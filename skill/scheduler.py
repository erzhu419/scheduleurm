#!/usr/bin/env python3
"""Multi-resource scheduler across local (4060 8GB / 16c / 64GB) + jtl110gpu
(2x 3080Ti 12GB / 12c / auto-probed RAM) + jtl110gpu2 (same) + jtl110cpu/jtl110cpu2
(Windows CPU-only / 128 physical cores / 512GB each).

Resource model: each task declares cpu_cores, ram_mb, vram_mb (or vram=0 for CPU-only). Placement requires
ALL three to fit on the chosen node + GPU. Per-task resource needs are auto-learned from history (peak
VRAM and peak RAM, cores user-declared) so re-runs of the same signature use accurate budgets.

Subcommands:
  submit    Add a task to the queue (no launch yet).
  dispatch  Probe nodes, pick placements for queued tasks, launch what fits. Same call doubles as rebalance.
  status    Show node telemetry + task table. Updates running-task health and peak VRAM/RAM.
  doctor    Audit queue invariants; --fix applies safe queued-task repairs.
  profile-local Run a local preflight directly and record resource/runtime history.
  results   Find inferred result artifacts in queue + archive.
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
import base64
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
from concurrent.futures import ThreadPoolExecutor, as_completed
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

_SKILL_DIR = Path(__file__).resolve().parent
_SCHEDULEURM_ROOT = _SKILL_DIR.parent
for _p in (str(_SKILL_DIR), str(_SCHEDULEURM_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from algorithm import load_placement_policy
except Exception:
    load_placement_policy = None

# ---------- node inventory ----------
JTL110GPU_RE_SAC_JAX_ENV = "/home/erzhu419/.venvs/resac-jax-gpu1-0438"
JTL110GPU_RE_SAC_JAX_SITE = f"{JTL110GPU_RE_SAC_JAX_ENV}/lib/python3.11/site-packages"
JTL110GPU_RE_SAC_JAX_NVIDIA_LIBS = ":".join([
    f"{JTL110GPU_RE_SAC_JAX_SITE}/nvidia/cublas/lib",
    f"{JTL110GPU_RE_SAC_JAX_SITE}/nvidia/cuda_nvcc/lib",
    f"{JTL110GPU_RE_SAC_JAX_SITE}/nvidia/cuda_nvrtc/lib",
    f"{JTL110GPU_RE_SAC_JAX_SITE}/nvidia/cuda_runtime/lib",
    f"{JTL110GPU_RE_SAC_JAX_SITE}/nvidia/cudnn/lib",
    f"{JTL110GPU_RE_SAC_JAX_SITE}/nvidia/cufft/lib",
    f"{JTL110GPU_RE_SAC_JAX_SITE}/nvidia/curand/lib",
    f"{JTL110GPU_RE_SAC_JAX_SITE}/nvidia/cusolver/lib",
    f"{JTL110GPU_RE_SAC_JAX_SITE}/nvidia/cusparse/lib",
    f"{JTL110GPU_RE_SAC_JAX_SITE}/nvidia/nccl/lib",
    f"{JTL110GPU_RE_SAC_JAX_SITE}/nvidia/nvjitlink/lib",
])
JTL110GPU_RE_SAC_JAX_PATH = (
    f"{JTL110GPU_RE_SAC_JAX_SITE}/nvidia/cuda_nvcc/bin:"
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)
OFFLINE_SUMO_ENV = "/home/erzhu419/.conda/envs/offline-sumo"
OFFLINE_SUMO_PYTHON = f"{OFFLINE_SUMO_ENV}/bin/python"
BUS_TORCH_CMD_REWRITES = [
    ("/home/erzhu419/anaconda3/bin/python -u sac_ensemble",
     f"{OFFLINE_SUMO_PYTHON} -u sac_ensemble"),
    ("/home/erzhu419/anaconda3/bin/python sac_ensemble",
     f"{OFFLINE_SUMO_PYTHON} sac_ensemble"),
]

# cpu_cores are the schedulable budget. ram_mb is an optional schedulable cap:
# positive value = cap placement to that many MB; 0/unset = auto-detect MemTotal.
# Local is WSL2: 16 physical cores / 32 logical threads. Use physical cores as
# the default schedulable budget; only count hyperthreads if intentionally oversubscribing.
# RAM headroom can be a fixed
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
    # remote=None means no concurrency cap (12 cores schedulable, probed RAM headroom is the bound).
    # max_vram_per_task: None = auto-derive from probed GPU total_mb at runtime (set in probe_node).
    # Was hardcoded 4096 from AMD 610 era; after NVIDIA 4060 (8188MB) became the default GPU,
    # the static cap silently blocked single-task allocations >4GB even though physically OK.
    "local":      {"host": None,         "cpu_cores": 16, "reserved_cpu_cores": int(os.environ.get("SCHEDULEURM_LOCAL_RESERVED_CPU_CORES", "0")),
                   "ram_mb": 56 * 1024,  "ram_headroom_mb": 2048, "ram_headroom_frac": 0.20, "max_vram_per_task": None, "max_concurrent_running": 10, "max_tasks_per_gpu": 3, "gpu_util_saturation_pct": None,
                   "capabilities": ["cpu", "cuda", "torch_cuda", "jax_cuda"],
                   "cmd_rewrites": BUS_TORCH_CMD_REWRITES},
    # jtl110gpu has a 525 driver, so the shared resac-jax JAX 0.9 CUDA plugin
    # cannot initialize there. Route RE-SAC JAX commands through a node-local
    # JAX 0.4.38 venv with CUDA/cuDNN wheels verified on this driver.
    "jtl110gpu":  {"host": "jtl110gpu",  "cpu_cores": 12, "ram_mb": 0, "ram_headroom_frac": 0.10, "max_vram_per_task": None, "max_concurrent_running": None, "max_tasks_per_gpu": 4, "enable_claims": True, "gpu_util_saturation_pct": None,
                   "capabilities": ["cpu", "cuda", "torch_cuda", "jax_cuda"],
                   "cmd_rewrites": BUS_TORCH_CMD_REWRITES + [
                       ("/home/erzhu419/.conda/envs/resac-jax/bin/python",
                        f"{JTL110GPU_RE_SAC_JAX_ENV}/bin/python"),
                   ],
                   "launch_extra_env": {
                       "PATH": JTL110GPU_RE_SAC_JAX_PATH,
                       "LD_LIBRARY_PATH": JTL110GPU_RE_SAC_JAX_NVIDIA_LIBS,
                       "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.20",
                   }},
    # Slurm may be installed on this small node, but the default policy is still
    # scheduleurm-managed local placement so one GPU can hold multiple small jobs.
    # Explicit task Slurm fields or per-node slurm_*_backend="slurm" still opt in.
    "jtl110gpu2": {"host": "jtl110gpu2", "cpu_cores": 12, "ram_mb": 0, "ram_headroom_frac": 0.10, "max_vram_per_task": None, "max_concurrent_running": None, "max_tasks_per_gpu": 4, "enable_claims": True, "gpu_util_saturation_pct": None,
                   "capabilities": ["cpu", "cuda", "torch_cuda", "jax_cuda"],
                   "cmd_rewrites": BUS_TORCH_CMD_REWRITES + [
                       ("/home/erzhu419/.conda/envs/resac-jax/bin/python",
                        f"{JTL110GPU_RE_SAC_JAX_ENV}/bin/python"),
                   ],
                   "launch_extra_env": {
                       "PATH": JTL110GPU_RE_SAC_JAX_PATH,
                       "LD_LIBRARY_PATH": JTL110GPU_RE_SAC_JAX_NVIDIA_LIBS,
                       "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.20",
                   }},
    # Campus-only HPC login node. Local Codex cannot reach it directly; route
    # all SSH/rsync/Slurm control traffic through GPU2, which is on campus net.
    # Workspaces are staged under the HPC account's home rather than preserving
    # /home/erzhu419, which that account cannot create.
    "zhengliang-hpc": {"host": "202.197.46.16", "ssh_user": "zhengliang01",
                       "ssh_proxy_jump": "jtl110gpu2",
                       "cpu_cores": 64, "ram_mb": 0, "ram_headroom_frac": 0.10,
                       "max_vram_per_task": None, "max_concurrent_running": None,
                       "slurm_backend": "slurm", "slurm_auto_large": True,
                       "slurm_auto_gpu_count": 4,
                       "slurm_gpu_partition": "gpu",
                       "slurm_cpu_partition": "cpu",
                       "slurm_gpu_details": {
                           "node007": {"model": "GeForce RTX 2080 Ti", "memory_gb": 11},
                       },
                       "only_when_targeted": True,
                       "stage_only_when_targeted": True,
                       "relay_node": "jtl110gpu2",
                       "relay_root": "/tmp/scheduleurm-hpc-relay/zhengliang-hpc",
                       "remote_workspace_root": "/home/zhengliang01/scheduleurm_work",
                       "remote_path_prefixes": [
                           str(Path.home() / "mine_code"),
                           "/home/erzhu419/mine_code",
                       ],
                       "capabilities": ["cpu", "cuda", "torch_cuda", "jax_cuda"]},
    # Direct compute-node mode for node007. The cluster's normal sshd PAM policy
    # rejects ordinary zhengliang01 SSH sessions on compute nodes unless Slurm
    # owns an allocation, but zndx can sudo-ssh to node007 as root. Run commands
    # through that root hop and immediately drop back to zhengliang01 so files and
    # experiment processes stay user-owned while scheduleurm can pack several
    # small RL jobs per physical GPU.
    "node007-direct": {"host": "202.197.46.16", "ssh_user": "zhengliang01",
                       "ssh_proxy_jump": "jtl110gpu2",
                       "sudo_ssh_host": "node007",
                       "sudo_ssh_run_as": "zhengliang01",
                       "cpu_cores": 64, "ram_mb": 0, "ram_headroom_frac": 0.10,
                       "max_vram_per_task": None, "max_concurrent_running": 16,
                       "max_tasks_per_gpu": 4,
                       "allow_gpu_over_one_third": True,
                       "gpu_util_saturation_pct": None,
                       "ignore_cpu_for_gpu_tasks": True,
                       "enable_claims": False,
                       "skip_launch_staging": True,
                       "only_when_targeted": True,
                       "stage_only_when_targeted": True,
                       "relay_node": "jtl110gpu2",
                       "relay_root": "/tmp/scheduleurm-hpc-relay/node007-direct",
                       "remote_workspace_root": "/home/zhengliang01/scheduleurm_work",
                       "remote_path_prefixes": [
                           str(Path.home() / "mine_code"),
                           "/home/erzhu419/mine_code",
                       ],
                       "cmd_rewrites": BUS_TORCH_CMD_REWRITES + [
                           ("/home/erzhu419/.conda/envs/resac-jax/bin/python",
                            "/home/zhengliang01/scheduleurm_work/conda_envs/resac-jax-535-py310-final/bin/python"),
                       ],
                       "launch_extra_env": {
                           "PATH": "/cm/local/apps/cuda-driver/libs/535.261.03/bin:/home/zhengliang01/scheduleurm_work/conda_envs/resac-jax-535-py310-final/bin:/usr/local/bin:/usr/bin:/bin",
                           "LD_LIBRARY_PATH": "/cm/local/apps/cuda-driver/libs/535.261.03/lib64",
                           "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.20",
                       },
                       "resume_scan_python": "/home/zhengliang01/scheduleurm_work/conda_envs/resac-jax-535-py310-final/bin/python",
                       "nvidia_smi_path": "/cm/local/apps/cuda-driver/libs/535.261.03/bin/nvidia-smi",
                       "capabilities": ["cpu", "cuda", "torch_cuda", "jax_cuda"]},
    # Windows CPU-only box. It has 256 logical / 128 physical cores split across
    # processor groups; WindowsBackend launches through a Python wrapper that
    # periodically pins child worker processes to unique physical cores. GPU
    # tasks never fit here because the probe reports gpus=[].
    "jtl110cpu":  {"host": "tf290q6n.zjz-service.cn", "ssh_user": "erzhu419", "ssh_port": 22945,
                   "ssh_identity": str(Path.home() / ".ssh" / "id_ed25519"),
                   "os": "windows", "cpu_cores": 128, "ram_mb": 512 * 1024,
                   "ram_headroom_frac": 0.10, "max_vram_per_task": 0,
                   "windows_python": r"F:\v\Scripts\python.exe",
                   "windows_workspace_root": r"F:\erzhu419_smoke",
                   "windows_scheduleurm_dir": r"F:\erzhu419_smoke\.scheduleurm",
                   "windows_auto_pin": True, "windows_skip_ht_pair": True,
                   "capabilities": ["cpu", "jax_cpu"],
                   "cpu_labor_node": True},
    "jtl110cpu2": {"host": "tf290q6n.zjz-service.cn", "ssh_user": "erzhu419", "ssh_port": 23565,
                   "ssh_identity": str(Path.home() / ".ssh" / "id_ed25519"),
                   "os": "windows", "cpu_cores": 128, "ram_mb": 512 * 1024,
                   "ram_headroom_frac": 0.10, "max_vram_per_task": 0,
                   "windows_python": r"F:\v\Scripts\python.exe",
                   "windows_workspace_root": r"F:\erzhu419_smoke",
                   "windows_scheduleurm_dir": r"F:\erzhu419_smoke\.scheduleurm",
                   "windows_auto_pin": True, "windows_skip_ht_pair": False,
                   "capabilities": ["cpu", "jax_cpu"],
                   "cpu_labor_node": True},
}

STATE_DIR = Path.home() / ".claude" / "scheduler"
QUEUE_FILE = STATE_DIR / "queue.json"
VRAM_FILE = STATE_DIR / "vram_history.json"  # holds {sig: {vram, ram, cpu}} (back-compat: int = vram only)
RUNTIME_FILE = STATE_DIR / "runtime_history.json"  # exact-parameter walltime/unit-time history
LOCK_FILE = STATE_DIR / ".lock"
LOG_DIR = STATE_DIR / "logs"

VRAM_MARGIN_MB = 500       # headroom on a GPU after placing a task
RAM_HEADROOM_FRAC = 0.10   # keep 10% of node RAM unallocated as buffer for OS/other procs
ONE_THIRD_PACK_RULE = True # don't add to a GPU already past the 1/3+grace freeze line
ONE_THIRD_PACK_GRACE_MB = int(os.environ.get("SCHEDULEURM_ONE_THIRD_PACK_GRACE_MB", "512"))
GPU_EVICT_ROLLBACK_MAX_AGE_S = int(os.environ.get("SCHEDULEURM_GPU_EVICT_ROLLBACK_MAX_AGE_S", "1800"))
GPU_EVICT_STABLE_PROGRESS_MIN_AGE_S = int(os.environ.get("SCHEDULEURM_GPU_EVICT_STABLE_PROGRESS_MIN_AGE_S", "900"))
RAM_HEADROOM_EVICTION_GRACE_MB = int(os.environ.get("SCHEDULEURM_RAM_HEADROOM_EVICTION_GRACE_MB", "512"))
RAM_EVICT_CKPT_PROTECT_MIN_AGE_S = int(os.environ.get("SCHEDULEURM_RAM_EVICT_CKPT_PROTECT_MIN_AGE_S", "300"))
RAM_EVICT_CKPT_PROTECT_PROGRESS = float(os.environ.get("SCHEDULEURM_RAM_EVICT_CKPT_PROTECT_PROGRESS", "0.01"))
GPU_UTIL_SATURATION_PCT = 85  # if an occupied GPU is past this compute util, don't pack more (would just contend)
DEFAULT_VRAM_MB = 512      # est for unknown signatures (no history, no siblings).
GPU_EMPTY_USED_MB = 200    # treat driver/runtime noise below this as an empty GPU
                           # Optimistic-low by design: if a task actually needs more, the
                           # post-dispatch resource-pressure rollback re-queues newly-launched
                           # colocated work before stable progress-bearing tasks. Once a card crosses the 1/3+grace
                           # freeze line, new GPU work is frozen until old work frees memory.
                           # Was 4096 → 1024 → 512. User passes --vram N when known larger.
DEFAULT_RAM_MB = 4096      # ditto for RAM
DEFAULT_CPU_CORES = 1      # most ML jobs are single-process at the dispatch level (workers are forked)

_PLACEMENT_POLICY = None
_PLACEMENT_POLICY_NAME = ""
_PLACEMENT_POLICY_LOAD_ERROR = ""


def _configure_algorithm(name: Optional[str] = None):
    """Load the optional placement policy.

    `legacy` is behavior-preserving.  Non-legacy policies live under sibling
    algorithm/ and are selected by env or CLI, so experiments can switch
    algorithms without editing scheduler.py.
    """
    global _PLACEMENT_POLICY, _PLACEMENT_POLICY_NAME, _PLACEMENT_POLICY_LOAD_ERROR
    explicit = bool(name)
    selected = (
        name
        or os.environ.get("SCHEDULEURM_ALGORITHM")
        or os.environ.get("SCHEDULEURM_PLACEMENT_POLICY")
        or "legacy"
    )
    if load_placement_policy is None:
        selected = "legacy"
        policy = None
    else:
        try:
            policy = load_placement_policy(selected)
            _PLACEMENT_POLICY_LOAD_ERROR = ""
        except Exception as e:
            if explicit:
                raise SystemExit(f"invalid --algorithm {selected!r}: {e}") from e
            _PLACEMENT_POLICY_LOAD_ERROR = str(e)
            policy = load_placement_policy("legacy")
    _PLACEMENT_POLICY = policy
    _PLACEMENT_POLICY_NAME = getattr(policy, "name", selected) if policy is not None else "legacy"
    return policy


def _algorithm_runtime_context() -> dict:
    return {
        "gpu_empty_used_mb": GPU_EMPTY_USED_MB,
        "vram_margin_mb": VRAM_MARGIN_MB,
        "one_third_pack_rule": ONE_THIRD_PACK_RULE,
        "one_third_pack_grace_mb": ONE_THIRD_PACK_GRACE_MB,
        "gpu_util_saturation_pct": GPU_UTIL_SATURATION_PCT,
    }


def _algorithm_name() -> str:
    if _PLACEMENT_POLICY is None:
        return "legacy"
    return getattr(_PLACEMENT_POLICY, "name", "legacy")


def _algorithm_config_snapshot() -> dict:
    policy = _PLACEMENT_POLICY
    if policy is None or not hasattr(policy, "snapshot"):
        return {"name": "legacy"}
    try:
        out = policy.snapshot()
        if _PLACEMENT_POLICY_LOAD_ERROR:
            out["load_error"] = _PLACEMENT_POLICY_LOAD_ERROR
        return out
    except Exception:
        return {"name": _algorithm_name(), "snapshot_error": True}


def _algorithm_gpu_fit_block_reason(task: dict, gpu: dict, node_info: dict) -> str:
    policy = _PLACEMENT_POLICY
    if policy is None:
        return ""
    try:
        return str(policy.gpu_fit_block_reason(
            task, gpu, node_info, _algorithm_runtime_context()) or "")
    except Exception as e:
        return f"algorithm:{_algorithm_name()}: error {str(e)[:120]}"


def _algorithm_gpu_score(task: dict, node_state: dict, gpu: dict, legacy_score):
    policy = _PLACEMENT_POLICY
    if policy is None:
        return legacy_score
    try:
        return policy.gpu_score(
            task, node_state, gpu, legacy_score, _algorithm_runtime_context())
    except Exception:
        return legacy_score


def _algorithm_selected_gpu_audit(task: dict, node_state: dict, gpu: dict) -> dict:
    policy = _PLACEMENT_POLICY
    if policy is None or not hasattr(policy, "selected_gpu_audit"):
        return {}
    try:
        return policy.selected_gpu_audit(
            task, node_state, gpu, _algorithm_runtime_context()) or {}
    except Exception as e:
        return {"error": str(e)[:120], "algorithm": _algorithm_name()}


_configure_algorithm()

ARCHIVE_FILE = STATE_DIR / "queue_archive.jsonl"
ARCHIVE_AGE_DAYS = 7        # terminal tasks older than this move from queue.json into archive
WATCHER_LOG_MAX_MB = 50     # detailed resource telemetry needs a longer JSONL tail
WATCHER_LOG_GENERATIONS = 3 # keep .log + .log.1 + .log.2 + .log.3 (oldest dropped on next rotation)
RESOURCE_LOG_INTERVAL_S = max(30, int(os.environ.get("SCHEDULEURM_RESOURCE_LOG_INTERVAL_S", "300")))
WINDOWS_WRAPPER_RESOURCE_LOG_INTERVAL_S = max(10, int(os.environ.get("SCHEDULEURM_WINDOWS_WRAPPER_RESOURCE_LOG_INTERVAL_S", "60")))
RUNNING_PROBE_MIN_INTERVAL_S = max(0, int(os.environ.get("SCHEDULEURM_RUNNING_PROBE_MIN_INTERVAL_S", "60")))
ETA_REFRESH_MIN_INTERVAL_S = max(0, int(os.environ.get("SCHEDULEURM_ETA_REFRESH_MIN_INTERVAL_S", "120")))
MAX_AUTO_RETRY = 3          # auto-requeue cap after crash (parent.retry_count + 1 > this → give up)
MAX_LAUNCH_RETRY = 3        # launch-failure cap (cwd missing, ssh timeout, etc.) before terminal failed + heal
LAUNCHING_RESET_S = 60      # stale WAL launch marker age before reverting to queued
NODE_DOWN_REQUEUE_S = 300   # opt-in reroute after backend probe is unknown this long
ESCALATIONS_FILE = STATE_DIR / "escalations.jsonl"  # /scheduler-heal reads this; watcher appends
WINDOWS_CPU_MAX_WORKERS_PER_PROCESS = int(os.environ.get(
    "SCHEDULEURM_WINDOWS_CPU_MAX_WORKERS_PER_PROCESS", "60"))

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

# Hardware-aware Slurm auto policy. Slurm remains a capability, not the default
# for small private boxes: only genuinely large/shared nodes plus large jobs get
# routed there automatically. Operators can still force either side per node.
SLURM_AUTO_MIN_CPU_CORES = int(os.environ.get("SCHEDULEURM_SLURM_AUTO_MIN_CPU_CORES", "128"))
SLURM_AUTO_MIN_GPUS = int(os.environ.get("SCHEDULEURM_SLURM_AUTO_MIN_GPUS", "8"))
SLURM_AUTO_LARGE_TASK_MIN_GPUS = int(os.environ.get("SCHEDULEURM_SLURM_AUTO_LARGE_TASK_MIN_GPUS", "2"))
SLURM_AUTO_LARGE_TASK_VRAM_MB = int(os.environ.get("SCHEDULEURM_SLURM_AUTO_LARGE_TASK_VRAM_MB", "20000"))
SLURM_AUTO_LARGE_TASK_CPU_CORES = int(os.environ.get("SCHEDULEURM_SLURM_AUTO_LARGE_TASK_CPU_CORES", "32"))

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

def dedup_claims(claims):
    """Phase 3.4.16 P1 fix: prior `claim` op blindly appended a new record
    without removing existing (scheduler_id, task_id) entries. So re-launch /
    migration / heal-resubmit produced multiple claim records for the same
    task — inflating apparent VRAM usage on whichever GPUs the previous
    attempts targeted (real-world incident: jtl110gpu2:GPU1 showed 2140MB
    used while nvidia-smi reported 10MB; 2 stale claims for tasks that had
    since moved to GPU0 had never been cleared).

    `claim` op is now upsert (see remote-script main()). This helper does
    the equivalent on every load so that pre-fix duplicate records still
    sitting in claims.json self-heal on the next op. Keep the LATEST
    record per (scheduler_id, task_id) by claimed_at; older duplicates
    are dropped.
    """
    by_key = {}
    for c in claims:
        key = (c.get("scheduler_id"), c.get("task_id"))
        prev = by_key.get(key)
        if prev is None:
            by_key[key] = c
            continue
        try:
            cur_ts = float(c.get("claimed_at") or 0)
            prev_ts = float(prev.get("claimed_at") or 0)
        except (TypeError, ValueError):
            cur_ts = prev_ts = 0
        if cur_ts > prev_ts:
            by_key[key] = c
    return list(by_key.values())

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
    used_cpu = sum(
        0 if c.get("ignore_cpu_capacity") else c.get("cpu_cores", 0)
        for c in active_claims
    )
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
    conflicts = []
    ignore_cpu = bool(payload.get("ignore_cpu_capacity"))
    if not ignore_cpu and used_cpu + cpu_need > cpu_cap:
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
        # 1/3 packing rule. Match local _gpu_fits semantics:
        #   - empty/noise-only GPU: allow the first task even if it is large
        #   - occupied GPU: don't add work if it is already at the freeze line
        #     OR this claim would cross it. The line includes a small grace
        #     window because 1/3 is a heuristic, not a physical OOM boundary.
        # There is deliberately no small-task exemption.
        if third_rule and not bool(payload.get("ignore_one_third_pack_rule")) and gcap > 0:
            third = gcap // 3
            grace = max(0, int(capacity.get("one_third_grace_mb", 0) or 0))
            freeze = min(gcap, third + grace)
            if gused > 100 and (gused >= freeze or gused + vram_need >= freeze):
                conflicts.append(
                    "gpu%s: occupied claim would cross 1/3+grace line "
                    "(claimed %dMB + need %dMB ≥ %dMB of %dMB, packing rule)" % (
                        gkey, gused, vram_need, freeze, gcap))
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
    claimed_cpu = sum(
        0 if c.get("ignore_cpu_capacity") else int(c.get("cpu_cores") or 0)
        for c in running_claims
    )
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
    # Phase 3.4.16 P1 fix: self-heal pre-fix duplicate (scheduler_id, task_id)
    # records left over from earlier claim() that didn't upsert. See
    # dedup_claims docstring for the bug + repro.
    fresh = dedup_claims(fresh)
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
        # Phase 3.4.16 P1 fix: claim is now UPSERT by (scheduler_id, task_id).
        # Pre-fix this was a blind append, which produced duplicate records
        # across re-launch / migration / heal-resubmit (each adding a new
        # entry, none clearing the prior one). Result: phantom VRAM usage
        # on stale GPU indices that the task had since moved away from.
        # Removing all prior records for the same (sid, tid) before the
        # append makes claim() idempotent and migration-safe.
        fresh = [c for c in fresh if record_key(c) != key]
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
        records = payload.get("records") or {}
        if not isinstance(records, dict):
            records = {}
        renewed = 0
        updated = 0
        for c in fresh:
            if c.get("scheduler_id") == sid and c.get("task_id") in tids:
                if new_exp:
                    c["expires_at"] = float(new_exp)
                rec = records.get(str(c.get("task_id")))
                if isinstance(rec, dict):
                    for key in ("gpu_idx", "vram_mb", "cpu_cores", "ram_mb", "pid",
                                "ignore_cpu_capacity", "ignore_one_third_pack_rule"):
                        if key not in rec:
                            continue
                        val = rec.get(key)
                        try:
                            if key in ("ignore_cpu_capacity", "ignore_one_third_pack_rule"):
                                c[key] = bool(val)
                            elif key in ("gpu_idx", "pid"):
                                c[key] = int(val) if val is not None else None
                            else:
                                c[key] = max(0, int(val))
                        except Exception:
                            pass
                    updated += 1
                renewed += 1
        data["claims"] = fresh
        data["intents"] = fresh_intents
        save_to_fd(fd, data)
        print(json.dumps({"ok": True, "renewed": renewed,
                          "updated": updated,
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
        (per-task VRAM cap, VRAM margin, 1/3 packing rule). Without these,
        two schedulers with stale
        probes could BOTH pass `_gpu_fits` locally on a fresh GPU and BOTH
        succeed at claim() — then the second-running task would violate
        the 1/3 rule even though the first scheduler's claim was meant to
        prevent that. Util saturation isn't replicated (no shared util
        reading); local pick_placement still gates on it before claim is
        ever invoked.
        """
        info = NODES.get(node, {})
        declared_ram = int(info.get("ram_mb") or 0)
        probed_ram = int((node_state or {}).get("total_ram_mb") or 0)
        if declared_ram and probed_ram:
            ram_cap = min(declared_ram, probed_ram)
        else:
            ram_cap = probed_ram or declared_ram
        cap = {
            "cpu_cores": int(info.get("cpu_cores", 0)),
            "ram_mb": ram_cap,
            "gpu_vram_mb": {},
            # Policy fields for cross-scheduler enforcement.
            "max_vram_per_task": info.get("max_vram_per_task"),
            "vram_margin_mb": int(VRAM_MARGIN_MB),
            "third_pack_rule": bool(ONE_THIRD_PACK_RULE),
            "one_third_grace_mb": int(ONE_THIRD_PACK_GRACE_MB),
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
        ignore_cpu = _ignore_cpu_for_server_gpu_task(
            task, node_state=node_state, node_info=NODES.get(node, {}),
            node_name=node, gpu_idx=gpu_idx)
        ignore_one_third = _task_ignores_one_third_pack_rule(task, NODES.get(node, {}))
        record = {
            "owner": owner,
            "scheduler_id": cls.scheduler_id(),
            "task_id": task["id"],
            "gpu_idx": gpu_idx,
            "vram_mb": int(task.get("est_vram_mb") or 0),
            "cpu_cores": 0 if ignore_cpu else int(task.get("cpu_cores") or DEFAULT_CPU_CORES),
            "ram_mb": int(task.get("ram_mb") or DEFAULT_RAM_MB),
            "ignore_cpu_capacity": bool(ignore_cpu),
            "ignore_one_third_pack_rule": bool(ignore_one_third),
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
    def renew_many(cls, node: str, task_ids: list, records: Optional[dict] = None) -> int:
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
            "records": records or {},
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


def _claim_resource_record_for_task(task: dict) -> dict:
    """Current claim budget for a running task.

    Claims are the shared scheduler budget, not a one-time launch receipt. CPU/RAM
    estimates can change after launch as probes learn real usage, and active claims
    must follow or they will block/allow placement on stale numbers.
    """
    pids = task.get("remote_pids") or []
    current_vram = int(task.get("current_vram_mb") or 0)
    est_vram = int(task.get("est_vram_mb") or 0)
    current_ram = int(task.get("current_ram_mb") or 0)
    est_ram = int(task.get("ram_mb") or DEFAULT_RAM_MB)
    node = task.get("node")
    ignore_cpu = _ignore_cpu_for_server_gpu_task(
        task, node_info=NODES.get(node or "", {}), node_name=node,
        gpu_idx=task.get("gpu_idx"))
    ignore_one_third = _task_ignores_one_third_pack_rule(task, NODES.get(node or "", {}))
    return {
        "gpu_idx": task.get("gpu_idx"),
        "vram_mb": max(0, est_vram, current_vram),
        "cpu_cores": 0 if ignore_cpu else max(0, int(task.get("cpu_cores") or DEFAULT_CPU_CORES)),
        "ram_mb": max(0, est_ram, current_ram),
        "ignore_cpu_capacity": bool(ignore_cpu),
        "ignore_one_third_pack_rule": bool(ignore_one_third),
        "pid": int(pids[0]) if pids else None,
    }


def _release_task_claims_and_intents(task: dict, extra_nodes=None,
                                     exclude_nodes=None, clear_markers: bool = True) -> int:
    """Best-effort release of this scheduler's claim and pending FIFO intents.

    release() removes both a real claim and a pre-launch intent on the remote
    claims file. `claim_intent_nodes` exists because CLAIM_RACE clears
    task["node"] before the task returns to queued.
    """
    exclude = set(exclude_nodes or [])
    nodes = []
    for key in ("node", "last_node"):
        if task.get(key):
            nodes.append(task.get(key))
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


def load_archive_tasks(limit: Optional[int] = None):
    """Return archived task records from queue_archive.jsonl.

    Archive is append-only JSONL. Treat bad lines as skipped records rather
    than making `show`/`results` unusable because one archival write was torn.
    """
    if not ARCHIVE_FILE.exists():
        return []
    tasks = []
    try:
        with open(ARCHIVE_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if isinstance(rec, dict):
                    tasks.append(rec)
    except Exception:
        return tasks[-limit:] if limit and len(tasks) > limit else tasks
    if limit and len(tasks) > limit:
        return tasks[-limit:]
    return tasks


def _find_task_record(task_id: str, include_archive: bool = True):
    """Find a task in hot state first, then archive. Returns (task, source)."""
    state = load_state()
    for t in state.get("tasks", []):
        if t.get("id") == task_id:
            return t, "queue"
    if include_archive:
        found = None
        for t in load_archive_tasks():
            if t.get("id") == task_id:
                found = t
        if found is not None:
            return found, "archive"
    return None, ""


_RESULT_FILE_EXTS = frozenset({
    ".csv", ".json", ".jsonl", ".pkl", ".pickle", ".pt", ".pth", ".ckpt",
    ".npz", ".npy", ".parquet", ".feather", ".txt", ".log", ".yaml", ".yml",
})
_RESULT_FILE_FLAGS = frozenset({
    "--output", "--out", "--outfile", "--out-file", "--out_file",
    "--out-csv", "--out_csv", "--output-csv", "--output_csv", "--csv",
    "--out-json", "--out_json", "--output-json", "--output_json", "--json",
    "--metrics", "--metrics-file", "--metrics_file",
    "--log-file", "--log_file",
})
_RESULT_DIR_FLAGS = frozenset({
    "--result-dir", "--result_dir", "--results-dir", "--results_dir",
    "--save-dir", "--save_dir", "--save-root", "--save_root",
    "--log-dir", "--log_dir", "--logging-dir", "--logging_dir",
    "--tb-dir", "--tb_dir", "--tensorboard-dir", "--tensorboard_dir",
})


def _clean_result_path(raw: str) -> str:
    if raw is None:
        return ""
    p = str(raw).strip().strip("'\"`<>")
    p = re.sub(r"\x1b\[[0-9;]*m", "", p)  # ANSI color
    p = p.rstrip(".,;)]}")
    if not p or p.startswith("-"):
        return ""
    suffix = Path(p).suffix.lower()
    if (p.startswith(("/", "./", "../", "~/", "$HOME/"))
            or "/" in p
            or suffix in _RESULT_FILE_EXTS):
        return p
    return ""


def _resolve_task_path(task: dict, raw_path: str) -> str:
    p = _clean_result_path(raw_path)
    if not p:
        return ""
    if p.startswith("$HOME/"):
        p = "~/" + p[len("$HOME/"):]
    # Keep remote home-relative paths home-relative; expanding locally would
    # use the scheduler user's home even when the artifact is on a remote node.
    if p.startswith("~/"):
        return p
    if os.path.isabs(p):
        return os.path.normpath(p)
    cwd = task.get("cwd") or ""
    if cwd and cwd not in ("(unknown)", "."):
        return os.path.normpath(os.path.join(cwd, p))
    return os.path.normpath(p)


def _result_kind_for_path(path: str, preferred: str = "") -> str:
    if preferred in ("file", "dir"):
        return preferred
    return "file" if Path(path).suffix.lower() in _RESULT_FILE_EXTS else "dir"


def _add_result_artifact(artifacts: list, task: dict, raw_path: str,
                         kind: str = "", source: str = ""):
    path = _resolve_task_path(task, raw_path)
    if not path:
        return
    node = task.get("node") or task.get("last_node") or "unknown"
    rec = {
        "path": path,
        "node": node,
        "kind": _result_kind_for_path(path, kind),
        "source": source or "inferred",
    }
    key = (rec["node"], rec["path"])
    for old in artifacts:
        if (old.get("node"), old.get("path")) == key:
            if rec["source"] not in (old.get("source") or ""):
                old["source"] = ",".join(
                    x for x in [old.get("source"), rec["source"]] if x)
            if old.get("kind") == "dir" and rec["kind"] == "file":
                old["kind"] = "file"
            return
    artifacts.append(rec)


def _cmd_flag_values(tokens: list, flags: set) -> dict:
    vals = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        flag, eq, val = tok.partition("=")
        if eq and flag in flags:
            vals.setdefault(flag, []).append(val)
            i += 1
            continue
        if tok in flags and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if not nxt.startswith("-"):
                vals.setdefault(tok, []).append(nxt)
                i += 2
                continue
        i += 1
    return vals


def _cmd_token_variants(cmd: str) -> list:
    """Return shlex token views for a command, including simple shell -c bodies."""
    if not cmd:
        return []
    variants = []

    def add(tokens):
        if tokens and tokens not in variants:
            variants.append(tokens)

    try:
        add(shlex.split(cmd))
    except Exception:
        add(str(cmd).split())

    shells = {"bash", "sh", "dash", "zsh", "ksh"}
    for tokens in list(variants):
        shell_seen = False
        for i, tok in enumerate(tokens[:-1]):
            if os.path.basename(str(tok)) in shells:
                shell_seen = True
                continue
            if shell_seen and tok in ("-c", "-lc"):
                try:
                    add(shlex.split(tokens[i + 1]))
                except Exception:
                    add(str(tokens[i + 1]).split())
                break
    return variants


def _cmd_flag_values_from_cmd(cmd: str, flags: set) -> dict:
    vals = {}
    for tokens in _cmd_token_variants(cmd):
        for flag, items in _cmd_flag_values(tokens, flags).items():
            bucket = vals.setdefault(flag, [])
            for item in items:
                if item not in bucket:
                    bucket.append(item)
    return vals


def _result_artifacts_from_cmd(task: dict) -> list:
    cmd = task.get("cmd") or ""
    artifacts = []
    if not cmd or (cmd.startswith("(auto-adopted") and " --" not in cmd):
        return artifacts
    file_vals = _cmd_flag_values_from_cmd(cmd, _RESULT_FILE_FLAGS)
    dir_vals = _cmd_flag_values_from_cmd(cmd, _RESULT_DIR_FLAGS)
    for flag, vals in file_vals.items():
        for v in vals:
            _add_result_artifact(artifacts, task, v, "file", f"cmd:{flag}")
    for flag, vals in dir_vals.items():
        for v in vals:
            _add_result_artifact(artifacts, task, v, "dir", f"cmd:{flag}")

    # Common ML convention: --save_root ROOT --run_name NAME writes under ROOT/NAME.
    roots = []
    for k in ("--save-root", "--save_root"):
        roots.extend(dir_vals.get(k, []))
    run_names = []
    for k in ("--run-name", "--run_name", "--name"):
        run_names.extend(_cmd_flag_values_from_cmd(cmd, {k}).get(k, []))
    if roots and run_names:
        for root in roots:
            for rn in run_names:
                if rn and not rn.startswith("-"):
                    _add_result_artifact(
                        artifacts, task, os.path.join(root, rn), "dir",
                        "cmd:save_root+run_name",
                    )
                    # RE-SAC/bus-style runners keep final models and logs under
                    # ROOT/{model,logs,pic}/RUN_NAME rather than ROOT/RUN_NAME.
                    for subdir in ("model", "logs", "pic"):
                        _add_result_artifact(
                            artifacts, task, os.path.join(root, subdir, rn), "dir",
                            f"cmd:save_root+{subdir}+run_name",
                        )
    return artifacts


_FINAL_MODEL_SUCCESS_FILES = ("final_policy", "final_q", "final_norm")


def _save_root_run_name_pairs_from_cmd(task: dict) -> list:
    cmd = task.get("cmd") or ""
    if not cmd or (cmd.startswith("(auto-adopted") and " --" not in cmd):
        return []
    vals = _cmd_flag_values_from_cmd(
        cmd,
        {"--save-root", "--save_root", "--run-name", "--run_name", "--name"},
    )
    roots = []
    for k in ("--save-root", "--save_root"):
        roots.extend(vals.get(k, []))
    run_names = []
    for k in ("--run-name", "--run_name", "--name"):
        run_names.extend(vals.get(k, []))
    pairs = []
    for root in roots:
        if not root or str(root).startswith("-"):
            continue
        for rn in run_names:
            if rn and not str(rn).startswith("-"):
                pairs.append((str(root), str(rn)))
    return pairs


def _bash_path_arg(path: str) -> str:
    p = str(path)
    if p.startswith("~/"):
        rest = p[2:]
        if not rest:
            return '"$HOME"'
        return '"$HOME"/' + shlex.quote(rest)
    return shlex.quote(p)


def _final_model_file_groups_from_cmd(task: dict) -> list:
    groups = []
    for root, rn in _save_root_run_name_pairs_from_cmd(task):
        model_dir = _resolve_task_path(task, os.path.join(root, "model", rn))
        if not model_dir:
            continue
        groups.append((
            model_dir,
            [os.path.join(model_dir, name) for name in _FINAL_MODEL_SUCCESS_FILES],
        ))
    return groups


def _mtimes_match_task_window(task: dict, mtimes: list) -> bool:
    if len(mtimes) < len(_FINAL_MODEL_SUCCESS_FILES):
        return False
    started = float(task.get("started_at") or 0)
    if started > 0 and min(float(x) for x in mtimes) < started - 300:
        return False
    return True


def _terminal_final_model_success(task: dict) -> str:
    """Treat complete final model triplets as a success marker for runners
    that exit without printing one."""
    groups = _final_model_file_groups_from_cmd(task)
    if not groups:
        return ""
    node = task.get("node") or "local"
    node_info = NODES.get(node) or {}
    for _model_dir, files in groups:
        try:
            if node_info.get("host") is None:
                mtimes = []
                for path in files:
                    p = Path(os.path.expanduser(path))
                    if not p.is_file():
                        break
                    mtimes.append(int(p.stat().st_mtime))
                if _mtimes_match_task_window(task, mtimes):
                    return "final_model_files"
            elif _node_is_windows(node):
                win_files = [_windows_path_for_task(task, p) for p in files]
                arr = "@(" + ",".join(_ps_quote(p) for p in win_files) + ")"
                ps = rf'''
$paths = {arr}
$mtimes = @()
foreach ($p in $paths) {{
  if (-not (Test-Path -LiteralPath $p -PathType Leaf)) {{ exit 1 }}
  $mtime = (Get-Item -LiteralPath $p).LastWriteTimeUtc
  $mtimes += [int64]([DateTimeOffset]::new($mtime).ToUnixTimeSeconds())
}}
$mtimes | ConvertTo-Json -Compress
'''
                rc, out, _ = _run_windows_ps(node, ps, timeout=10, check=False)
                if rc != 0 or not (out or "").strip():
                    continue
                data = json.loads(out.strip().splitlines()[-1])
                mtimes = data if isinstance(data, list) else [data]
                if _mtimes_match_task_window(task, mtimes):
                    return "final_model_files"
            else:
                quoted = [_bash_path_arg(p) for p in files]
                tests = " && ".join(f"test -f {p}" for p in quoted)
                cmd = f"{tests} && stat -c %Y {' '.join(quoted)}"
                rc, out, _ = run_on(node, cmd, timeout=10, check=False)
                if rc != 0 or not (out or "").strip():
                    continue
                mtimes = [int(x) for x in out.split() if x.strip().isdigit()]
                if _mtimes_match_task_window(task, mtimes):
                    return "final_model_files"
        except Exception:
            continue
    return ""


_LOG_RESULT_RE = re.compile(
    r"(?i)(?:"
    r"results?\s+saved\s+to|logging\s+to|"
    r"output(?:\s+written)?(?:\s+to)?|"
    r"output\s+already\s+present|"
    r"out_(?:json|csv|path)|"
    r"saved(?:\s+\w+){0,4}\s+to|saved"
    r")\s*:?\s+(?P<path>\S+)"
)


def _fetch_log_result_lines(task: dict, max_lines: int = 50) -> list:
    """Return log lines that look like they mention result paths.

    Tail-only scans miss scripts that print `Output: ...` at startup and then
    run for hours. Use a streaming local scan or remote grep to extract only
    candidate lines.
    """
    log_path = task.get("log_path")
    if not log_path or task.get("auto_adopted"):
        return []
    node = task.get("node")
    grep_re = (
        r"Results saved to|Logging to|Output|output already present|"
        r"out_json|out_csv|Saved"
    )
    try:
        if node and NODES.get(node, {}).get("host") is None:
            lp = Path(log_path)
            if not lp.exists():
                return []
            r = subprocess.run(
                ["grep", "-E", "-i", "-m", str(int(max_lines)), grep_re, str(lp)],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode not in (0, 1):
                return []
            return [l for l in (r.stdout or "").splitlines() if _LOG_RESULT_RE.search(l)]
        elif node and _node_is_windows(node):
            win_path = _windows_path_for_task(task, log_path)
            ps = rf'''
$path = {_ps_quote(win_path)}
if (-not (Test-Path -LiteralPath $path)) {{ exit 0 }}
Get-Content -LiteralPath $path -ErrorAction SilentlyContinue |
  Select-String -Pattern {_ps_quote(grep_re)} |
  Select-Object -First {int(max_lines)} |
  ForEach-Object {{ $_.Line }}
'''
            rc, out, _ = _run_windows_ps(node, ps, timeout=15, check=False)
            if rc != 0 or not out:
                return []
            return [l for l in out.splitlines() if _LOG_RESULT_RE.search(l)]
        elif node:
            rc, out, _ = run_on(
                node,
                f"grep -E -i -m {int(max_lines)} {shlex.quote(grep_re)} "
                f"{shlex.quote(log_path)} 2>/dev/null || true",
                timeout=15, check=False,
            )
            if rc != 0 or not out:
                return []
            return [l for l in out.splitlines() if _LOG_RESULT_RE.search(l)]
    except Exception:
        return []
    return []


def _result_artifacts_from_log(task: dict) -> list:
    artifacts = []
    tail_text, _ = _fetch_log_tail(task)
    if not tail_text:
        diag = task.get("_diagnosis") or {}
        tail_text = diag.get("tail") or ""
    lines = []
    if tail_text and tail_text != "(no log)":
        lines.extend(tail_text.splitlines())
    lines.extend(_fetch_log_result_lines(task))
    if not lines:
        return artifacts
    for line in lines:
        m = _LOG_RESULT_RE.search(line)
        if not m:
            continue
        raw = m.group("path")
        kind = "file" if Path(_clean_result_path(raw)).suffix.lower() in _RESULT_FILE_EXTS else ""
        _add_result_artifact(artifacts, task, raw, kind, "log")
    return artifacts


def _discover_result_artifacts(task: dict, include_log: bool = True) -> list:
    artifacts = []
    if task.get("result_dir"):
        _add_result_artifact(artifacts, task, task.get("result_dir"), "dir", "declared:result_dir")
    for rec in _result_artifacts_from_cmd(task):
        _add_result_artifact(artifacts, task, rec["path"], rec.get("kind"), rec.get("source"))
    if include_log:
        for rec in _result_artifacts_from_log(task):
            _add_result_artifact(artifacts, task, rec["path"], rec.get("kind"), rec.get("source"))
    return artifacts


def _record_result_artifacts(task: dict) -> list:
    """Persist best-effort result artifacts on a terminal task record."""
    discovered = _discover_result_artifacts(task, include_log=True)
    if not discovered:
        return []
    combined = []
    for rec in (task.get("result_artifacts") or []) + discovered:
        _add_result_artifact(combined, task, rec.get("path"), rec.get("kind"), rec.get("source"))
    task["result_artifacts"] = combined
    task["result_artifacts_discovered_at"] = time.time()
    return combined

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

def _task_duration_s(task):
    if task.get("started_at") and task.get("finished_at"):
        return max(0, int(task["finished_at"] - task["started_at"]))
    return 0

def _task_has_progress_marker(task):
    """True once a task has made real train/eval progress, not just initialized."""
    if task.get("runtime_current_unit") or task.get("progress_ratio"):
        return True
    line = task.get("last_progress_line") or ""
    return bool(re.search(r"\bIter\s+\d+\b|\[\s*\d+\s*/\s*\d+\]|Epoch\s+\d+|%\|", line))

def _is_oom_like_task(task):
    if task.get("failure_category") == "OOM":
        return True
    text = f"{task.get('last_block_reason') or ''}\n{(task.get('_diagnosis') or {}).get('reason') or ''}\n{(task.get('_diagnosis') or {}).get('tail') or ''}".lower()
    return "out of memory" in text or "resource_exhausted" in text

def _untrusted_startup_oom_sample(task, duration_s=None):
    """Short OOM before progress is usually JAX/CUDA prealloc failure, not steady-state need.

    Example: a JAX task without XLA_PYTHON_CLIENT_PREALLOCATE=false can briefly grab most
    of a 12GB card during graph instantiation, fail after ~50s, and leave a 9GB peak. Using
    that as future est_vram pins otherwise small RE-SAC tasks out of the queue.
    """
    if task.get("status") != "failed":
        return False
    if not _is_oom_like_task(task):
        return False
    if _task_has_progress_marker(task):
        return False
    duration_s = _task_duration_s(task) if duration_s is None else duration_s
    return 0 < duration_s <= 5 * 60

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
            if peak > 100 and not _untrusted_startup_oom_sample(t):
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
            if peak > 100 and not _untrusted_startup_oom_sample(t):
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

def _live_sibling_ram_floor(task, state):
    """Return a RAM floor from currently-running sibling tasks, or 0.

    This is intentionally narrower than _effective_est_ram's full project fallback. It is used
    to raise stale queued RAM estimates when live siblings prove the old estimate is too low.
    For broad project-level matches, require at least two live samples so one large outlier
    cannot poison a mixed project.
    """
    project = task.get("project")
    if not project:
        return 0
    sig = task.get("signature") or ""
    prefix = "/".join(sig.split("/")[:2]) if sig else ""
    desc_key = (task.get("description") or "").split(":")[0].strip().lower()
    self_id = task.get("id")
    desc_candidates = []
    prefix_candidates = []
    project_candidates = []
    for t in state.get("tasks", []):
        if t.get("id") == self_id:
            continue
        if t.get("project") != project:
            continue
        if t.get("status") not in ("running", "launching"):
            continue
        ram = max(int(t.get("current_ram_mb") or 0), int(t.get("peak_ram_mb") or 0))
        if ram <= 100:
            continue
        project_candidates.append(ram)
        td = (t.get("description") or "").split(":")[0].strip().lower()
        if desc_key and td == desc_key:
            desc_candidates.append(ram)
        tsig = t.get("signature") or ""
        if prefix and "/".join(tsig.split("/")[:2]) == prefix:
            prefix_candidates.append(ram)

    for candidates, min_samples in (
        (desc_candidates, 1),
        (prefix_candidates, 1),
        (project_candidates, 2),
    ):
        if len(candidates) >= min_samples:
            candidates.sort()
            return int(candidates[len(candidates) // 2])
    return 0

HISTORY_MAX_ENTRIES = int(os.environ.get("SCHEDULEURM_HISTORY_MAX_ENTRIES", "5000"))
HISTORY_SAMPLES_PER_SIG = 10
HISTORY_PERCENTILE = 80   # p80 of last N samples → estimate
RUNTIME_HISTORY_MAX_ENTRIES = int(os.environ.get("SCHEDULEURM_RUNTIME_HISTORY_MAX_ENTRIES", "5000"))
RUNTIME_HISTORY_SAMPLES_PER_KEY = 10
RUNTIME_HISTORY_PERCENTILE = 80
RUNTIME_WALLTIME_MULT = 1.20
RUNTIME_MIN_WALLTIME_S = 10 * 60  # progress-derived walltime can be shorter than legacy 1h
RUNTIME_CLOSEST_MIN_SCORE = 0.58

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
def _node_is_windows(node: str) -> bool:
    return str((NODES.get(node) or {}).get("os") or "").lower().startswith("win")


def _ssh_base_args(node: str) -> list:
    """Build the ssh argv prefix for a node.

    Existing GPU nodes use ~/.ssh/config aliases (`host=jtl110gpu`). Windows
    CPU nodes can instead declare host/user/port directly so they do not need
    an alias entry. BatchMode stays on: the watcher must be unattended, so
    jtl110cpu needs key-based SSH even if the manual recipe once used a
    password.
    """
    info = NODES[node]
    host = info["host"]
    args = [
        "ssh",
        "-o", "ConnectTimeout=5",
        "-o", "ServerAliveInterval=5",
        "-o", "ServerAliveCountMax=3",
        "-o", "BatchMode=yes",
    ]
    if info.get("ssh_identity"):
        args.extend(["-i", os.path.expanduser(str(info["ssh_identity"]))])
    if info.get("ssh_port"):
        args.extend(["-p", str(info["ssh_port"])])
    proxy_jump = info.get("ssh_proxy_jump") or info.get("proxy_jump")
    if proxy_jump:
        args.extend(["-J", str(proxy_jump)])
    for opt in (info.get("ssh_options") or []):
        opt = str(opt or "").strip()
        if opt:
            args.extend(["-o", opt])
    user = info.get("ssh_user")
    target = f"{user}@{host}" if user and "@" not in str(host) else str(host)
    args.append(target)
    return args


def _ssh_no_stdin_args(node: str) -> list:
    """Build ssh argv for commands that must not consume scheduler stdin.

    Relay commands can run another ssh/rsync/sbatch on the remote host. If the
    outer ssh keeps stdin open, an inner ssh may drain the parent script's stdin
    and skip later commands. Use this for ordinary remote shell commands only.
    """
    args = _ssh_base_args(node)
    return args[:1] + ["-n"] + args[1:]


def _run_windows_ps(node: str, ps_script: str, timeout=15, check=True, input_data: bytes | None = None):
    """Run a PowerShell script on a Windows node over OpenSSH."""
    encoded = base64.b64encode((ps_script or "").encode("utf-16le")).decode("ascii")
    base_cmd = _ssh_base_args(node) + [
        "powershell",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy", "Bypass",
    ]
    if len(encoded) < 7000:
        proc = subprocess.run(
            base_cmd + [
                "-EncodedCommand", encoded,
            ],
            input=input_data,
            capture_output=True, timeout=timeout,
        )
    else:
        if input_data is not None:
            raise RuntimeError("cannot pass stdin to oversized Windows PowerShell script")
        # Windows command lines cap out around 8 KiB; larger scheduler probes
        # must travel over stdin to a temporary script instead of directly in
        # -EncodedCommand. The short runner still uses EncodedCommand so SSH
        # quoting stays predictable.
        runner = r'''
$p = Join-Path $env:TEMP ("scheduleurm_" + [guid]::NewGuid().ToString() + ".ps1")
$enc = New-Object System.Text.UTF8Encoding $false
[IO.File]::WriteAllText($p, [Console]::In.ReadToEnd(), $enc)
& powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $p
$ec = $LASTEXITCODE
Remove-Item -LiteralPath $p -Force -ErrorAction SilentlyContinue
if ($null -eq $ec) { $ec = 0 }
exit ([int]$ec)
'''
        runner_encoded = base64.b64encode(runner.encode("utf-16le")).decode("ascii")
        proc = subprocess.run(
            base_cmd + [
                "-EncodedCommand", runner_encoded,
            ],
            input=(ps_script or "").encode("utf-8"),
            capture_output=True, timeout=timeout,
        )
    stdout = (proc.stdout or b"").decode("utf-8", "replace")
    stderr = (proc.stderr or b"").decode("utf-8", "replace")
    if check and proc.returncode != 0:
        raise RuntimeError(f"[{node}] powershell failed (rc={proc.returncode}): {stderr.strip()[:300]}")
    return proc.returncode, stdout, stderr


def _windows_probe_error_hint(node: str, msg: str) -> str:
    text = (msg or "").strip()
    if "Permission denied" in text or "publickey,password" in text:
        ident = (NODES.get(node, {}) or {}).get("ssh_identity") or "~/.ssh/id_ed25519"
        return (f"{text[:180]} | SSH key auth failed for Windows node {node}; "
                f"install {os.path.expanduser(str(ident))}.pub on the host. "
                "Do not store the password in scheduler.py.")
    if "timed out" in text.lower() or "TimeoutExpired" in text:
        return (f"{text[:180]} | Windows probe timed out; check SSH reachability, "
                "PowerShell startup, and host CPU pressure.")
    return text[:300]


def _windows_tail_ps(path: str, max_bytes: int = 4096) -> str:
    """PowerShell snippet that prints the last max_bytes of a text log.

    Use FileShare.ReadWrite because scheduler-owned logs are written by a
    still-running wrapper process. Get-Content can block or fail on those.
    """
    max_bytes = max(1, int(max_bytes))
    return rf'''
$p = {_ps_quote(path)}
if (Test-Path -LiteralPath $p) {{
  $fs = [IO.File]::Open($p, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
  try {{
    $len = [int64]$fs.Length
    $start = [Math]::Max([int64]0, $len - {max_bytes})
    [void]$fs.Seek($start, [IO.SeekOrigin]::Begin)
    $buf = New-Object byte[] ([int]($len - $start))
    $read = $fs.Read($buf, 0, $buf.Length)
    [Console]::OutputEncoding = [Text.Encoding]::UTF8
    [Text.Encoding]::UTF8.GetString($buf, 0, $read)
  }} finally {{
    $fs.Close()
  }}
}}
'''


def _windows_path_for_task(task: dict, path: str) -> str:
    """Map an absolute or cwd-relative task path onto the Windows node layout."""
    node = task.get("node")
    if not node or not _node_is_windows(node) or not path:
        return path
    text = str(path).strip().strip('"').strip("'")
    if re.match(r"^[A-Za-z]:[\\/]", text) or text.startswith("\\\\"):
        return text.replace("/", "\\")
    if text.startswith("/home/") or text.startswith(str(Path.home())):
        return _windows_path_for_node(node, text)
    cwd = task.get("cwd") or ""
    if cwd:
        base = _windows_path_for_node(node, cwd).rstrip("\\/")
        return base + "\\" + text.replace("/", "\\")
    return text.replace("/", "\\")


def _is_windows_native_path(path: str) -> bool:
    text = str(path or "").strip().strip('"').strip("'")
    return bool(re.match(r"^[A-Za-z]:[\\/]", text) or text.startswith("\\\\"))


def _fetch_windows_text_tail(task: dict, path: str, max_bytes: int = 4096) -> tuple[str, int]:
    node = task.get("node")
    if not node:
        return ("", 0)
    win_path = _windows_path_for_task(task, path)
    ps = _windows_tail_ps(win_path, max_bytes=max_bytes) + rf'''
Write-Output '___SZ___'
if (Test-Path -LiteralPath {_ps_quote(win_path)}) {{
  Write-Output ([int64](Get-Item -LiteralPath {_ps_quote(win_path)}).Length)
}} else {{
  Write-Output 0
}}
'''
    rc, out, _ = _run_windows_ps(node, ps, timeout=10, check=False)
    if rc != 0 or not out:
        return ("", 0)
    if "___SZ___" in out:
        body, _, sz_str = out.rpartition("___SZ___")
        try:
            return body, int((sz_str or "0").strip().splitlines()[-1])
        except Exception:
            return body, 0
    return out, 0


def _scan_windows_log_for_patterns(task: dict, patterns: list[str]) -> list[str]:
    node = task.get("node")
    log_path = task.get("log_path")
    if not node or not log_path:
        return []
    win_path = _windows_path_for_task(task, log_path)
    pats_b64 = base64.b64encode(json.dumps(patterns).encode("utf-8")).decode("ascii")
    ps = rf'''
$path = {_ps_quote(win_path)}
if (-not (Test-Path -LiteralPath $path)) {{ exit 0 }}
$patterns = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String({_ps_quote(pats_b64)})) | ConvertFrom-Json
$hits = @()
foreach ($pat in $patterns) {{
  if (Select-String -LiteralPath $path -SimpleMatch -CaseSensitive -Pattern $pat -Quiet -ErrorAction SilentlyContinue) {{
    $hits += [string]$pat
  }}
}}
$hits | ConvertTo-Json -Compress
'''
    rc, out, _ = _run_windows_ps(node, ps, timeout=15, check=False)
    if rc != 0 or not (out or "").strip():
        return []
    try:
        data = json.loads(out.strip().splitlines()[-1])
        if isinstance(data, str):
            return [data]
        if isinstance(data, list):
            return [str(x) for x in data]
    except Exception:
        return []
    return []


def run_on(node, shell_cmd, timeout=15, check=True):
    """Run a bash shell_cmd on a Unix-like node. Returns (returncode, stdout, stderr).

    Remote ssh argv is built by _ssh_base_args(), including
    ServerAliveInterval=5 and ServerAliveCountMax=3 for half-dead sockets.
    For ordinary commands it is wrapped with `ssh -n` so relay-side nested ssh
    calls cannot consume scheduler stdin.
    """
    if _node_is_windows(node):
        raise RuntimeError(f"run_on() bash path called for Windows node {node}; use _run_windows_ps")
    host = NODES[node]["host"]
    if host is None:
        proc = subprocess.run(["bash", "-lc", shell_cmd], capture_output=True, timeout=timeout, text=True)
    else:
        remote_cmd = _remote_bash_command_for_node(node, shell_cmd, timeout=timeout)
        # ServerAlive*: detect dead ControlMaster sockets fast (item 9 — half-dead persistent
        # connection used to take the full subprocess timeout to surface). 5s × 3 missed pings
        # = 15s before ssh declares dead; under our 15s default `timeout=`, this means a
        # broken master surfaces as ssh failure not subprocess timeout.
        proc = subprocess.run(
            _ssh_no_stdin_args(node) + [remote_cmd],
            capture_output=True, timeout=timeout, text=True,
        )
    if check and proc.returncode != 0:
        raise RuntimeError(f"[{node}] cmd failed (rc={proc.returncode}): {proc.stderr.strip()[:300]}")
    return proc.returncode, proc.stdout, proc.stderr


def _remote_bash_command_for_node(node: str, shell_cmd: str, timeout: Optional[int] = None) -> str:
    """Build the command executed by the outer SSH target.

    Most Unix nodes execute the command directly on the SSH target. Some campus
    compute nodes are reachable only by first SSHing to a login node and then
    sudo-SSHing to the compute node; for those, run the command through the
    inner root hop and optionally drop to the configured user before executing
    the actual bash payload.
    """
    info = NODES.get(node, {}) or {}
    inner_host = str(info.get("sudo_ssh_host") or "").strip()
    if not inner_host:
        return f"bash -lc {shlex.quote(shell_cmd)}"
    # Avoid login shells here. The HPC account's login profile prints Slurm
    # status tables to stdout, which corrupts machine-readable probes such as
    # nvidia-smi CSV and claim-manager JSON.
    inner = f"bash -c {shlex.quote(shell_cmd)}"
    run_as = str(info.get("sudo_ssh_run_as") or "").strip()
    if run_as and run_as != "root":
        inner = f"su {shlex.quote(run_as)} -s /bin/bash -c {shlex.quote(inner)}"
    ssh_bin = str(info.get("sudo_ssh_bin") or "/usr/bin/ssh")
    # zndx already has a narrow sudoers rule shaped like:
    #   zhengliang01 ALL=(root) NOPASSWD: /usr/bin/ssh node007, /usr/bin/ssh node007 *
    # Keep the host immediately after ssh_bin so that rule matches. Stdin is
    # detached via shell redirection rather than `ssh -n`, which would change
    # argv and miss the sudoers command pattern.
    inner_ssh_args = [ssh_bin, inner_host, inner]
    sudo_cmd = "sudo -n " + " ".join(shlex.quote(arg) for arg in inner_ssh_args) + " </dev/null"
    try:
        remote_timeout = max(1, int(timeout) - 1) if timeout else 0
    except Exception:
        remote_timeout = 0
    wrapped = f"timeout -k 2s {remote_timeout}s {sudo_cmd}" if remote_timeout else sudo_cmd
    return f"bash -c {shlex.quote(wrapped)}"

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
      host_free_ram_mb         — Windows available physical memory
      host_total_ram_mb        — Windows total physical memory
      host_cpu_load_pct        — Windows host CPU load percentage
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
    # Keep host RAM/CPU separate from GPU counters. The GPU Engine counter can
    # stall for several seconds under load; host RAM is cheap and should still
    # be reported so TUI/status can explain WSL-vs-Windows memory differences.
    script = (
        "try { "
        "  Add-Type -AssemblyName Microsoft.VisualBasic -ErrorAction Stop; "
        "  $ci = New-Object Microsoft.VisualBasic.Devices.ComputerInfo; "
        "  $free = [math]::Round($ci.AvailablePhysicalMemory / 1MB); "
        "  $tot  = [math]::Round($ci.TotalPhysicalMemory / 1MB); "
        "} catch { "
        "  $os = Get-CimInstance Win32_OperatingSystem; "
        "  $free = [math]::Round($os.FreePhysicalMemory / 1024); "
        "  $tot  = [math]::Round($os.TotalVisibleMemorySize / 1024); "
        "} "
        "$cpu = Get-CimInstance Win32_Processor -ErrorAction SilentlyContinue "
        "| Measure-Object -Property LoadPercentage -Average; "
        "$cpu_pct = if ($cpu.Average -ne $null) { [math]::Round($cpu.Average) } else { 0 }; "
        "Write-Output \"$free|$tot|$cpu_pct\""
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
        if len(bits) >= 2:
            try:
                out["host_free_ram_mb"] = int(bits[0])
                out["host_total_ram_mb"] = int(bits[1])
                if len(bits) >= 3:
                    out["host_cpu_load_pct"] = int(bits[2])
            except ValueError:
                pass
    except Exception:
        pass
    gpu_script = (
        "$gpu = (Get-Counter '\\GPU Engine(*engtype_Compute*)\\Utilization "
        "Percentage' -ErrorAction SilentlyContinue).CounterSamples "
        "| Measure-Object -Maximum CookedValue; "
        "$gpu_pct = if ($gpu.Maximum) { [math]::Round($gpu.Maximum) } else { 0 }; "
        "Write-Output \"$gpu_pct\""
    )
    try:
        r = subprocess.run(
            [pwsh, "-NoProfile", "-NonInteractive", "-Command", gpu_script],
            capture_output=True, text=True, timeout=max(1.0, min(3.0, timeout_s)),
        )
        if r.returncode == 0:
            line = (r.stdout or "").strip().splitlines()[-1] if r.stdout else ""
            if line:
                out["gpu_compute_util_pct"] = int(float(line))
    except Exception:
        pass
    return out


def _probe_windows_node(name: str) -> dict:
    """Probe a Windows CPU-only node.

    Windows has no /proc/loadavg and jtl110cpu has no GPU. Use host-wide CPU
    load, then translate it into the observed schedulable physical-core budget.
    The static config is only an upper cap: after BIOS / hardware changes a node
    may expose fewer logical CPUs than it used to.
    """
    def _schedulable_cpu_from_logical(logical_cores: int, info: dict) -> int:
        try:
            declared = int(info.get("cpu_cores") or 0)
        except Exception:
            declared = 0
        try:
            logical = int(logical_cores or 0)
        except Exception:
            logical = 0
        observed = 0
        if logical > 0:
            observed = logical
            if info.get("windows_skip_ht_pair", True) and logical > 1:
                observed = max(1, logical // 2)
        if declared > 0 and observed > 0:
            return max(1, min(declared, observed))
        return max(1, declared or observed or 1)

    def _cheap_observed_cpu_capacity(info: dict) -> tuple[int, int]:
        """Return (schedulable_physical, observed_logical) without WMI/CIM."""
        cap = _cheap_host_capacity(info)
        if cap.get("logical_cpu"):
            logical = int(cap["logical_cpu"])
            return _schedulable_cpu_from_logical(logical, info), logical
        total = _schedulable_cpu_from_logical(0, info)
        return total, 0

    def _cheap_host_capacity(info: dict, timeout_s: float = 8.0) -> dict:
        """Return cheap Windows host capacity counters without WMI/CIM.

        The high-core-count Windows boxes can be so CPU-saturated that CIM and
        Get-Counter calls time out. Microsoft.VisualBasic.Devices.ComputerInfo
        plus registry CPU counting has stayed responsive under that load, so use
        it as the RAM/core source even when the richer probe falls back to queue
        accounting for CPU load.
        """
        ps = r'''
$logical = [int][System.Environment]::ProcessorCount
try {
  $envLogical = [int]$env:NUMBER_OF_PROCESSORS
  if ($envLogical -gt $logical) { $logical = $envLogical }
} catch {}
try {
  $regCount = (Get-ChildItem 'HKLM:\HARDWARE\DESCRIPTION\System\CentralProcessor' -ErrorAction Stop | Measure-Object).Count
  if ($regCount -gt $logical) { $logical = [int]$regCount }
} catch {}
try {
  Add-Type -AssemblyName Microsoft.VisualBasic | Out-Null
  $ci = New-Object Microsoft.VisualBasic.Devices.ComputerInfo
  $free = [int][math]::Round($ci.AvailablePhysicalMemory / 1MB)
  $total = [int][math]::Round($ci.TotalPhysicalMemory / 1MB)
} catch {
  $free = 0
  $total = 0
}
Write-Output "$free|$total|$logical"
'''
        try:
            rc, out, _ = _run_windows_ps(name, ps, timeout=timeout_s, check=False)
            if rc == 0:
                for line in reversed((out or "").splitlines()):
                    line = line.strip()
                    if re.match(r"^\d+\|\d+\|\d+$", line):
                        free_s, total_s, logical_s = line.split("|", 2)
                        return {
                            "free_ram_mb": int(free_s),
                            "total_ram_mb": int(total_s),
                            "logical_cpu": int(logical_s),
                        }
        except Exception:
            pass
        return {}

    def _windows_queue_usage() -> tuple[int, int]:
        used_cpu = 0
        used_ram = 0
        try:
            raw = json.loads(QUEUE_FILE.read_text())
            for t in raw.get("tasks", []):
                if t.get("status") != "running" or t.get("node") != name:
                    continue
                used_cpu += max(0, int(t.get("cpu_cores") or 0))
                used_ram += max(0, int(t.get("current_ram_mb") or t.get("ram_mb") or 0))
        except Exception:
            pass
        return used_cpu, used_ram

    def _queue_accounting_fallback(reason: str) -> dict:
        """Fallback when live Windows counters time out under heavy CPU load.

        Windows PowerShell/CIM can become too slow exactly when the box is full.
        Reporting the node as DOWN in that case is also wrong: active scheduler
        tasks are still alive, and the TUI/dispatcher should at least see the
        scheduler-owned CPU budget already placed on that host. Verify a cheap
        SSH command first so true network outages still show DOWN.
        """
        try:
            ping = subprocess.run(
                _ssh_base_args(name) + ["cmd", "/c", "echo", "OK"],
                capture_output=True, timeout=6,
            )
            if ping.returncode != 0 or b"OK" not in (ping.stdout or b""):
                return {"name": name, "alive": False,
                        "error": _windows_probe_error_hint(name, reason)}
        except Exception:
            return {"name": name, "alive": False,
                    "error": _windows_probe_error_hint(name, reason)}
        info = NODES.get(name, {})
        total_cpu, observed_logical = _cheap_observed_cpu_capacity(info)
        cap = _cheap_host_capacity(info)
        declared_ram = int(info.get("ram_mb") or 0)
        probed_total_ram = int(cap.get("total_ram_mb") or 0)
        actual_free_ram = int(cap.get("free_ram_mb") or 0)
        if declared_ram and probed_total_ram:
            total_ram = min(declared_ram, probed_total_ram)
        else:
            total_ram = declared_ram or probed_total_ram
        used_cpu, used_ram = _windows_queue_usage()
        queue_free_ram = max(0, total_ram - used_ram) if total_ram else 0
        sched_free_ram = (
            min(actual_free_ram, total_ram) if actual_free_ram and total_ram
            else queue_free_ram
        )
        return {
            "name": name,
            "alive": True,
            "gpus": [],
            "free_ram_mb": max(0, sched_free_ram),
            "actual_free_ram_mb": actual_free_ram,
            "total_ram_mb": total_ram,
            "free_cpu": max(0, total_cpu - used_cpu),
            "total_cpu": total_cpu,
            "loadavg": float(min(total_cpu, used_cpu)),
            "cpu_load_pct": int(round(100.0 * min(total_cpu, used_cpu) / max(1, total_cpu))),
            "cores": observed_logical,
            "logical_cpu": observed_logical,
            "physical_cpu": total_cpu,
            "os": "windows",
            "probe_fallback": "queue_accounting",
            "probe_fallback_reason": str(reason)[:200],
        }

    info = NODES.get(name, {})
    cpu_load_source = str(
        info.get("windows_cpu_load_source")
        or os.environ.get("SCHEDULEURM_WINDOWS_CPU_LOAD_SOURCE")
        or "queue"
    ).strip().lower()
    if cpu_load_source in ("queue", "accounting", "queue_accounting"):
        cap = _cheap_host_capacity(info)
        if cap:
            observed_logical = int(cap.get("logical_cpu") or 0)
            total_cpu = _schedulable_cpu_from_logical(observed_logical, info)
            declared_ram = int(info.get("ram_mb") or 0)
            probed_total_ram = int(cap.get("total_ram_mb") or 0)
            actual_free_ram = int(cap.get("free_ram_mb") or 0)
            if declared_ram and probed_total_ram:
                total_ram = min(declared_ram, probed_total_ram)
            else:
                total_ram = declared_ram or probed_total_ram or max(1, actual_free_ram)
            used_cpu, _used_ram = _windows_queue_usage()
            free_cpu = max(0, total_cpu - used_cpu)
            return {
                "name": name,
                "alive": True,
                "gpus": [],
                "free_ram_mb": min(actual_free_ram, total_ram) if total_ram else actual_free_ram,
                "actual_free_ram_mb": actual_free_ram,
                "total_ram_mb": total_ram,
                "free_cpu": free_cpu,
                "total_cpu": total_cpu,
                "loadavg": float(min(total_cpu, used_cpu)),
                "cpu_load_pct": int(round(100.0 * min(total_cpu, used_cpu) / max(1, total_cpu))),
                "cores": observed_logical,
                "logical_cpu": observed_logical,
                "physical_cpu": total_cpu,
                "os": "windows",
                "probe_fallback": "queue_cpu_live_ram",
            }

    script = r'''
$ErrorActionPreference = "Continue"
try {
  Add-Type -AssemblyName Microsoft.VisualBasic | Out-Null
  $ci = New-Object Microsoft.VisualBasic.Devices.ComputerInfo
  $free = [int][math]::Round($ci.AvailablePhysicalMemory / 1MB)
  $total = [int][math]::Round($ci.TotalPhysicalMemory / 1MB)
  $logical = [int][System.Environment]::ProcessorCount
  try {
    $envLogical = [int]$env:NUMBER_OF_PROCESSORS
    if ($envLogical -gt $logical) { $logical = $envLogical }
  } catch {}
  try {
    $regCount = (Get-ChildItem 'HKLM:\HARDWARE\DESCRIPTION\System\CentralProcessor' -ErrorAction Stop | Measure-Object).Count
    if ($regCount -gt $logical) { $logical = [int]$regCount }
  } catch {}
  try {
    $cs = Get-CimInstance Win32_ComputerSystem -ErrorAction SilentlyContinue
    if ($cs.NumberOfLogicalProcessors -and [int]$cs.NumberOfLogicalProcessors -gt $logical) { $logical = [int]$cs.NumberOfLogicalProcessors }
  } catch {}
  $loadPct = 0
  try {
    $cpu = Get-CimInstance Win32_Processor -ErrorAction SilentlyContinue | Measure-Object -Property LoadPercentage -Average
    if ($cpu.Average -ne $null) { $loadPct = [int][math]::Round($cpu.Average) }
  } catch { $loadPct = 0 }
  if ($loadPct -le 0) {
    try {
      $sample = (Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 1 -MaxSamples 1).CounterSamples[0].CookedValue
      $loadPct = [int][math]::Round($sample)
    } catch { $loadPct = 0 }
  }
} catch {
  $ErrorActionPreference = "Stop"
  $os = Get-CimInstance Win32_OperatingSystem
  $cs = Get-CimInstance Win32_ComputerSystem
  $cpu = Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average
  $free = [int][math]::Round($os.FreePhysicalMemory / 1024)
  $total = [int][math]::Round($os.TotalVisibleMemorySize / 1024)
  if ($cs.NumberOfLogicalProcessors -and [int]$cs.NumberOfLogicalProcessors -gt $logical) { $logical = [int]$cs.NumberOfLogicalProcessors }
  $loadPct = 0
  if ($cpu.Average -ne $null) { $loadPct = [int][math]::Round($cpu.Average) }
}
Write-Output "$free|$total|$logical|$loadPct"
'''
    try:
        rc, out, err = _run_windows_ps(name, script, timeout=6, check=False)
        if rc != 0:
            return {"name": name, "alive": False,
                    "error": _windows_probe_error_hint(name, err or out or "powershell probe failed")}
        lines = [ln.strip() for ln in (out or "").splitlines()]
        line = next(
            (ln for ln in reversed(lines)
             if re.match(r"^\d+(\.\d+)?\|\d+(\.\d+)?\|\d+(\.\d+)?\|\d+(\.\d+)?$", ln)),
            "",
        )
        if not line:
            raise ValueError((out or "no probe output").strip()[:120])
        free_s, total_s, logical_s, load_s = line.split("|")[:4]
        free_ram = int(float(free_s))
        probed_total_ram = int(float(total_s))
        logical_cores = int(float(logical_s))
        load_pct = max(0.0, min(100.0, float(load_s)))
    except Exception as e:
        return _queue_accounting_fallback(str(e))

    if free_ram <= 0 or probed_total_ram <= 0 or logical_cores <= 0:
        cap = _cheap_host_capacity(info)
        free_ram = free_ram or int(cap.get("free_ram_mb") or 0)
        probed_total_ram = probed_total_ram or int(cap.get("total_ram_mb") or 0)
        logical_cores = logical_cores or int(cap.get("logical_cpu") or 0)
    declared_ram = int(info.get("ram_mb") or 0)
    if declared_ram and probed_total_ram:
        total_ram = min(declared_ram, probed_total_ram)
    else:
        total_ram = declared_ram or probed_total_ram or max(1, free_ram)
    sched_free_ram = min(free_ram, total_ram)
    total_cpu = _schedulable_cpu_from_logical(logical_cores, info)
    used_cpu = int(round(total_cpu * load_pct / 100.0))
    free_cpu = max(0, total_cpu - used_cpu)
    return {
        "name": name,
        "alive": True,
        "gpus": [],
        "free_ram_mb": sched_free_ram,
        "actual_free_ram_mb": free_ram,
        "total_ram_mb": total_ram,
        "free_cpu": free_cpu,
        "total_cpu": total_cpu,
        "loadavg": float(used_cpu),
        "cpu_load_pct": int(load_pct),
        "cores": logical_cores,
        "logical_cpu": logical_cores,
        "physical_cpu": total_cpu,
        "os": "windows",
    }


def _parse_slurm_int(value, default: int = 0) -> int:
    try:
        return int(float(str(value or "").strip()))
    except Exception:
        return default


def _parse_slurm_tres(text: str) -> dict:
    out = {}
    for part in str(text or "").split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = _parse_slurm_int(v, 0)
    return out


def _parse_slurm_gres_gpu(gres: str, cfg_tres: str = "") -> int:
    gres = str(gres or "")
    if gres and gres != "(null)":
        total = 0
        for m in re.finditer(r"(?:^|,)gpu(?::[^:,]+)?:(\d+)", gres):
            total += _parse_slurm_int(m.group(1), 0)
        if total:
            return total
    return _parse_slurm_tres(cfg_tres).get("gres/gpu", 0)


def _format_slurm_gpu_memory_label(detail: dict) -> str:
    raw = detail.get("memory_label")
    if raw:
        return str(raw)
    raw = detail.get("memory_gb")
    if raw is not None:
        try:
            gb = float(raw)
            return f"{gb:g}GB"
        except Exception:
            return str(raw)
    raw = detail.get("memory_mb")
    if raw is not None:
        return _format_mem_gb(raw)
    return ""


def _parse_scontrol_node_line(line: str) -> Optional[dict]:
    if not line.startswith("NodeName="):
        return None
    fields = {m.group(1): m.group(2) for m in re.finditer(r"(\w+)=([^ ]+)", line)}
    if not fields.get("NodeName"):
        return None
    total_cpu = _parse_slurm_int(fields.get("CPUTot"), 0)
    alloc_cpu = _parse_slurm_int(fields.get("CPUAlloc"), 0)
    real_mem = _parse_slurm_int(fields.get("RealMemory"), 0)
    alloc_mem = _parse_slurm_int(fields.get("AllocMem"), 0)
    free_mem = _parse_slurm_int(fields.get("FreeMem"), 0)
    total_gpus = _parse_slurm_gres_gpu(fields.get("Gres"), fields.get("CfgTRES"))
    alloc_gpus = _parse_slurm_tres(fields.get("AllocTRES")).get("gres/gpu", 0)
    state = (fields.get("State") or "UNKNOWN").split("+", 1)[0]
    parts = [p for p in (fields.get("Partitions") or "").split(",") if p and p != "(null)"]
    return {
        "name": fields.get("NodeName"),
        "partitions": parts,
        "state": state,
        "gres": fields.get("Gres") or "(null)",
        "total_cpu": total_cpu,
        "alloc_cpu": alloc_cpu,
        "free_cpu": max(0, total_cpu - alloc_cpu),
        "loadavg": float(fields.get("CPULoad") or 0.0),
        "real_mem_mb": real_mem,
        "alloc_mem_mb": alloc_mem,
        "sched_free_mem_mb": max(0, real_mem - alloc_mem) if real_mem else max(0, free_mem),
        "host_free_mem_mb": free_mem,
        "total_gpus": total_gpus,
        "alloc_gpus": alloc_gpus,
        "free_gpus": max(0, total_gpus - alloc_gpus),
    }


def _partition_state_counts(nodes: list[dict]) -> str:
    counts = {}
    for rec in nodes:
        st = str(rec.get("state") or "UNKNOWN").lower()
        counts[st] = counts.get(st, 0) + 1
    return ",".join(f"{k}:{counts[k]}" for k in sorted(counts))


def _probe_slurm_cluster_node(name: str) -> dict:
    """Probe real Slurm compute resources instead of login-node CPU/RAM.

    For campus clusters reached through a login node, /proc and nvidia-smi on
    the SSH target describe the login host. The schedulable resources live in
    Slurm's controller, so TUI/status should render partition/node/GRES data.
    """
    cmd = (
        "LC_ALL=C sinfo -h -o 'SINFO|%P|%D|%t|%C|%G|%m|%N' 2>/dev/null; "
        "echo __SCHED_SCONTROL__; "
        "LC_ALL=C scontrol show node -o 2>/dev/null"
    )
    try:
        rc, out, err = run_on(name, cmd, timeout=20, check=False)
    except Exception as e:
        return {"name": name, "alive": False, "error": str(e)[:160]}
    if rc != 0:
        return {"name": name, "alive": False, "error": (err or out or f"slurm probe rc={rc}")[:160]}

    node_recs = []
    for line in (out or "").splitlines():
        line = line.strip()
        rec = _parse_scontrol_node_line(line)
        if rec:
            node_recs.append(rec)
    if not node_recs:
        return {"name": name, "alive": False, "error": "slurm probe returned no compute nodes"}

    gpu_details = (NODES.get(name, {}) or {}).get("slurm_gpu_details") or {}
    for rec in node_recs:
        detail = gpu_details.get(rec.get("name")) or {}
        if not isinstance(detail, dict):
            continue
        model = detail.get("model") or detail.get("gpu_model") or detail.get("name")
        mem_label = _format_slurm_gpu_memory_label(detail)
        if model:
            rec["gpu_model"] = str(model)
        if mem_label:
            rec["gpu_mem_label"] = mem_label

    parts: dict[str, list[dict]] = {}
    for rec in node_recs:
        for part in rec.get("partitions") or ["unknown"]:
            parts.setdefault(part, []).append(rec)

    part_summaries = []
    for part, recs in sorted(parts.items(), key=lambda kv: (0 if kv[0] == "gpu" else 1, kv[0])):
        total_cpu = sum(int(r.get("total_cpu") or 0) for r in recs)
        free_cpu = sum(int(r.get("free_cpu") or 0) for r in recs)
        total_mem = sum(int(r.get("real_mem_mb") or 0) for r in recs)
        free_mem = sum(int(r.get("sched_free_mem_mb") or 0) for r in recs)
        total_gpus = sum(int(r.get("total_gpus") or 0) for r in recs)
        free_gpus = sum(int(r.get("free_gpus") or 0) for r in recs)
        gpu_recs = [r for r in recs if int(r.get("total_gpus") or 0) > 0]
        gpu_model_values = sorted({
            str(r.get("gpu_model"))
            for r in gpu_recs
            if r.get("gpu_model")
        })
        gpu_mem_values = sorted({
            str(r.get("gpu_mem_label"))
            for r in gpu_recs
            if r.get("gpu_mem_label")
        })
        gpu_model = ""
        if gpu_recs and len(gpu_model_values) == 1 and all(r.get("gpu_model") for r in gpu_recs):
            gpu_model = gpu_model_values[0]
        gpu_mem_label = ""
        if gpu_recs and len(gpu_mem_values) == 1 and all(r.get("gpu_mem_label") for r in gpu_recs):
            gpu_mem_label = gpu_mem_values[0]
        part_summaries.append({
            "name": part,
            "nodes": len(recs),
            "node_names": [r.get("name") for r in recs],
            "states": _partition_state_counts(recs),
            "total_cpu": total_cpu,
            "free_cpu": free_cpu,
            "total_mem_mb": total_mem,
            "free_mem_mb": free_mem,
            "total_gpus": total_gpus,
            "free_gpus": free_gpus,
            "gpu_model": gpu_model,
            "gpu_mem_label": gpu_mem_label,
        })

    total_cpu = sum(int(r.get("total_cpu") or 0) for r in node_recs)
    free_cpu = sum(int(r.get("free_cpu") or 0) for r in node_recs)
    total_mem = sum(int(r.get("real_mem_mb") or 0) for r in node_recs)
    free_mem = sum(int(r.get("sched_free_mem_mb") or 0) for r in node_recs)
    total_gpus = sum(int(r.get("total_gpus") or 0) for r in node_recs)
    free_gpus = sum(int(r.get("free_gpus") or 0) for r in node_recs)
    return {
        "name": name,
        "alive": True,
        "gpus": [],
        "slurm_cluster": True,
        "slurm_nodes": node_recs,
        "slurm_partitions": part_summaries,
        "total_gpus": total_gpus,
        "free_gpus": free_gpus,
        "free_cpu": free_cpu,
        "total_cpu": total_cpu,
        "loadavg": sum(float(r.get("loadavg") or 0.0) for r in node_recs),
        "free_ram_mb": free_mem,
        "actual_free_ram_mb": free_mem,
        "total_ram_mb": total_mem,
        "probe_source": "slurm",
    }


def _format_slurm_cluster_summary(n: dict) -> str:
    parts = []
    for p in n.get("slurm_partitions") or []:
        name = p.get("name") or "?"
        nodes = p.get("nodes") or 0
        node_names = ",".join(p.get("node_names") or [])
        node_part = f"{nodes} node" + ("" if nodes == 1 else "s")
        if node_names and nodes <= 3:
            node_part = node_names
        bits = [
            f"{name}: {node_part}",
            f"cpu={p.get('free_cpu', 0)}/{p.get('total_cpu', 0)}",
            f"mem={_format_mem_gb(p.get('free_mem_mb', 0))}/{_format_mem_gb(p.get('total_mem_mb', 0))}",
        ]
        if int(p.get("total_gpus") or 0) > 0:
            gpu_bit = f"gpu={p.get('free_gpus', 0)}/{p.get('total_gpus', 0)}"
            gpu_detail = " ".join(
                str(x) for x in (
                    f"{p.get('gpu_mem_label')}/card" if p.get("gpu_mem_label") else "",
                    p.get("gpu_model") or "",
                ) if x
            )
            if gpu_detail:
                gpu_bit += f"({gpu_detail})"
            bits.insert(1, gpu_bit)
        if p.get("states"):
            bits.append(f"state={p.get('states')}")
        parts.append(" ".join(bits))
    return "slurm " + " ; ".join(parts) if parts else "slurm (no partitions)"


def probe_node(name):
    if _node_is_windows(name):
        return _probe_windows_node(name)
    info = NODES.get(name, {})
    if _slurm_mode_enabled(info.get("slurm_backend")) or info.get("probe_slurm_cluster"):
        return _probe_slurm_cluster_node(name)
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
    info = NODES.get(name, {})
    nvsmi = str(info.get("nvidia_smi_path") or info.get("nvidia_smi") or "nvidia-smi")
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
    # Positive ram_mb in NODES is an optional schedulable cap, not necessarily
    # the physical total. ram_mb=0/unset means use the probe directly.
    sched_free_ram = min(free_ram, total_ram) if total_ram else free_ram
    total_cpu = info.get("cpu_cores", cores) or cores or 1
    free_cpu = max(0, total_cpu - int(round(loadavg)))
    result = {"name": name, "alive": True, "gpus": gpus,
              "free_ram_mb": sched_free_ram, "actual_free_ram_mb": free_ram, "total_ram_mb": total_ram,
              "free_cpu": free_cpu, "total_cpu": total_cpu, "loadavg": loadavg,
              "cores": cores}
    # Phase 3.3/3.6: for WSL2 `local`, fold Windows-host metrics so the TUI can
    # show numbers that match what the user sees in Task Manager. CPU is not
    # display-only: Windows-side saturation still steals physical cores from WSL,
    # so local placement uses the more conservative of WSL loadavg and host CPU.
    if name == "local":
        extras = _probe_windows_host_extras()
        if extras:
            host_free = extras.get("host_free_ram_mb")
            if host_free is not None:
                try:
                    host_free = max(0, int(host_free))
                    result["wsl_free_ram_mb"] = result.get("free_ram_mb")
                    result["wsl_actual_free_ram_mb"] = result.get("actual_free_ram_mb")
                    result["wsl_total_ram_mb"] = result.get("total_ram_mb")
                    result["host_free_ram_mb"] = host_free
                    result["free_ram_mb"] = min(int(result.get("free_ram_mb") or 0), host_free)
                except Exception:
                    pass
            result["host_total_ram_mb"] = extras.get("host_total_ram_mb")
            host_cpu = extras.get("host_cpu_load_pct")
            if host_cpu is not None:
                host_cpu = max(0, min(100, int(host_cpu)))
                host_used = int(round(total_cpu * host_cpu / 100.0))
                result["host_cpu_load_pct"] = host_cpu
                result["host_cpu_used_cores"] = host_used
                result["wsl_free_cpu"] = result["free_cpu"]
                result["wsl_loadavg"] = loadavg
                result["free_cpu"] = min(result["free_cpu"], max(0, total_cpu - host_used))
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
        for g in n.get("gpus") or []:
            g.setdefault("observed_used_mb", int(g.get("used_mb") or 0))
            g.setdefault("observed_free_mb", int(g.get("free_mb") or 0))

        active_cpu = sum(
            0 if c.get("ignore_cpu_capacity") else int(c.get("cpu_cores") or 0)
            for c in active
        )
        active_ram = sum(int(c.get("ram_mb") or 0) for c in active)
        pending_cpu = sum(
            0 if c.get("ignore_cpu_capacity") else int(c.get("cpu_cores") or 0)
            for c in pending
        )
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
    if int(vram_mb or 0) > 0:
        task.pop("vram_estimation_source", None)
        task.pop("vram_estimation_note", None)


def _format_mem_gb(mb, *, approx: bool = False) -> str:
    """Format a memory quantity stored in MB as a compact GB string."""
    try:
        val = float(mb or 0) / 1024.0
    except Exception:
        val = 0.0
    digits = 2 if abs(val) < 10 else 1
    prefix = "~" if approx else ""
    return f"{prefix}{val:.{digits}f}GB"


def _format_node_ram_summary(n: dict) -> str:
    free = n.get("free_ram_mb")
    if free is None:
        return ""
    host_free = n.get("host_free_ram_mb")
    if host_free is not None:
        return f"ram_free={_format_mem_gb(free)}(eff)"
    return f"ram_free={_format_mem_gb(free)}"


def _local_user() -> str:
    try:
        import getpass as _getpass
        return _getpass.getuser()
    except Exception:
        return os.environ.get("USER") or "unknown"


def _local_host_short() -> str:
    try:
        import socket as _socket
        return (_socket.gethostname() or "host").split(".", 1)[0]
    except Exception:
        return "host"


def _actor_info(action: str = "", reason: str = "") -> dict:
    try:
        scheduler_id = _ClaimManager.scheduler_id()
    except Exception:
        scheduler_id = ""
    user = _local_user()
    host = _local_host_short()
    return {
        "user": user,
        "host": host,
        "label": f"{user}@{host}",
        "scheduler_id": scheduler_id,
        "action": action,
        "reason": str(reason or "")[:500],
        "ts": time.time(),
    }


def _record_task_kill_actor(task: dict, action: str, reason: str) -> dict:
    actor = _actor_info(action, reason)
    task["last_killed_at"] = actor["ts"]
    task["last_killed_by"] = actor["label"]
    task["last_kill_actor"] = actor
    task["last_kill_action"] = action
    task["last_kill_reason"] = str(reason or "")[:500]
    return actor


def _format_task_owner(task: dict) -> str:
    """Human-facing owner/source label.

    Same Unix account can hide multiple real people on shared boxes. When a
    task was adopted rather than launched by this scheduler, mark it explicitly
    as external? even if the OS user is the same.
    """
    user = (task.get("submitted_by") or task.get("process_owner")
            or task.get("owner") or _local_user())
    origin = task.get("origin") or ""
    sid = task.get("scheduler_id") or task.get("submitted_scheduler_id") or ""
    try:
        own_sid = _ClaimManager.scheduler_id()
    except Exception:
        own_sid = ""
    if task.get("auto_adopted") or origin in ("external", "manual-adopt"):
        tag = "external?" if origin != "manual-adopt" else "manual"
    elif sid and own_sid and sid != own_sid:
        tag = "other-sched?"
    else:
        tag = "this"
    return f"{user}:{tag}"


def _ceil_div(a: int, b: int) -> int:
    a = int(a)
    b = max(1, int(b))
    return (a + b - 1) // b


def _cpu_labor_node_names() -> list:
    return [name for name, info in NODES.items() if info.get("cpu_labor_node")]


def _node_physical_cores(node: Optional[str], node_state: Optional[dict] = None) -> int:
    """Best estimate of schedulable physical cores for CPU-worker planning.

    Some probes report logical CPUs (2N with SMT), while scheduleurm's
    `cpu_cores` budget is normally already physical-core-sized. Prefer explicit
    physical_cores, then configured schedulable budget, then infer N from 2N.
    """
    info = NODES.get(node or "", {}) if node else {}
    node_state = node_state or {}
    for key in ("physical_cores", "physical_cpu", "physical_cpus"):
        try:
            v = int(info.get(key) or node_state.get(key) or 0)
            if v > 0:
                return v
        except Exception:
            pass
    try:
        sched = int(info.get("cpu_cores") or node_state.get("total_cpu") or 0)
    except Exception:
        sched = 0
    try:
        logical = int(node_state.get("logical_cpu") or node_state.get("logical_cores")
                      or node_state.get("cores") or 0)
    except Exception:
        logical = 0
    if sched > 0:
        return sched
    if logical > 1 and (info.get("windows_skip_ht_pair", True) or logical > 64):
        return max(1, logical // 2)
    return max(1, logical or DEFAULT_CPU_CORES)


def _cpu_planning_cores_for_node(node: str, node_state: Optional[dict] = None) -> tuple:
    """Return (available_physical_cores, total_physical_cores) for CPU batch planning."""
    total = _node_physical_cores(node, node_state)
    if isinstance(node_state, dict):
        if node_state.get("alive") is False:
            return 0, total
        if "free_cpu" in node_state:
            try:
                free = max(0, int(node_state.get("free_cpu") or 0))
                return min(total, free), total
            except Exception:
                return 0, total
    return total, total


def _cpu_worker_plan_for_items(total_items: int, physical_cores: int) -> dict:
    """Return the user's preferred worker plan for M independent CPU items.

    Given M items and N physical cores:
      e = ceil(M / N) waves
      f = ceil(M / e) workers
    This maximizes worker occupancy while avoiding more workers than physical
    cores, and makes all waves almost equal.
    """
    m = max(0, int(total_items or 0))
    n = max(1, int(physical_cores or 1))
    if m <= 0:
        return {"items": 0, "physical_cores": n, "waves": 0,
                "workers": 0, "last_wave_items": 0}
    waves = _ceil_div(m, n)
    workers = min(n, _ceil_div(m, waves))
    last = m - workers * max(0, waves - 1)
    if last <= 0:
        last = workers
    return {"items": m, "physical_cores": n, "waves": waves,
            "workers": workers, "last_wave_items": last}


def _cpu_max_workers_per_process(node: Optional[str]) -> int:
    """Per-process worker cap for runtime/platform limits.

    Windows multiprocessing uses WaitForMultipleObjects internally; one process
    cannot safely wait on 64+ worker handles. Keep a conservative cap and let
    CPU batch planning split high-worker node shards into multiple processes.
    """
    info = NODES.get(node or "", {}) if node else {}
    try:
        explicit = int(info.get("max_cpu_workers_per_process") or 0)
        if explicit > 0:
            return explicit
    except Exception:
        pass
    if str(info.get("os") or "").lower() == "windows":
        return max(1, WINDOWS_CPU_MAX_WORKERS_PER_PROCESS)
    return 0


def _cpu_batch_plan(total_items: int, node_names: Optional[list] = None,
                    node_states: Optional[dict] = None) -> list:
    """Split M independent CPU items across CPU-labor nodes by free physical cores."""
    m = max(0, int(total_items or 0))
    if m <= 0:
        return []
    names = list(node_names or _cpu_labor_node_names())
    if not names:
        names = ["local"] if "local" in NODES else list(NODES.keys())[:1]
    node_states = node_states or {}
    caps = []
    for name in names:
        avail, total = _cpu_planning_cores_for_node(name, node_states.get(name))
        if avail > 0:
            caps.append((name, avail, total))
    if not caps:
        return []
    total_cap = sum(cap for _, cap, _total in caps)
    global_waves = _ceil_div(m, total_cap)
    raw = []
    assigned = 0
    for idx, (name, cap, total) in enumerate(caps):
        share = m * cap / float(total_cap)
        base = int(share)
        raw.append([name, cap, total, base, share - base, idx])
        assigned += base
    remaining = m - assigned
    raw.sort(key=lambda r: (-r[4], r[5]))
    for i in range(remaining):
        raw[i % len(raw)][3] += 1
    raw.sort(key=lambda r: r[5])

    out = []
    cursor = 0
    active = [r for r in raw if r[3] > 0]
    node_shard_idx = 0
    for name, cap, total, items, _frac, _idx in active:
        start = cursor
        end = start + items
        cursor = end
        per_node = _cpu_worker_plan_for_items(items, cap)
        # Use the cluster-wide wave count when splitting across nodes so peers
        # finish in the same number of rounds when capacities are comparable.
        workers = min(cap, _ceil_div(items, global_waves))
        waves = _ceil_div(items, workers) if workers else 0
        last = items - workers * max(0, waves - 1)
        if last <= 0 and items > 0:
            last = workers
        base = {
            "node": name,
            "physical_cores": cap,
            "available_cores": cap,
            "total_physical_cores": total,
            "items": items,
            "start": start,
            "end": end,
            "workers": workers,
            "waves": waves,
            "single_node_workers": per_node["workers"],
            "single_node_waves": per_node["waves"],
            "last_wave_items": last,
            "global_waves": global_waves,
            "node_shard_index": node_shard_idx,
        }
        node_shard_idx += 1
        max_workers = _cpu_max_workers_per_process(name)
        if max_workers and workers > max_workers:
            lanes = _ceil_div(workers, max_workers)
            lane_cursor = start
            remaining = items
            for lane_idx in range(lanes):
                lane_count = lanes - lane_idx
                lane_items = _ceil_div(remaining, lane_count)
                lane_start = lane_cursor
                lane_end = lane_start + lane_items
                lane_cursor = lane_end
                remaining -= lane_items
                lane_workers = min(max_workers, _ceil_div(lane_items, max(1, waves)))
                if lane_items > 0 and lane_workers <= 0:
                    lane_workers = 1
                lane_waves = _ceil_div(lane_items, lane_workers) if lane_workers else 0
                lane_last = lane_items - lane_workers * max(0, lane_waves - 1)
                if lane_last <= 0 and lane_items > 0:
                    lane_last = lane_workers
                lane = dict(base)
                lane.update({
                    "items": lane_items,
                    "start": lane_start,
                    "end": lane_end,
                    "workers": lane_workers,
                    "waves": lane_waves,
                    "last_wave_items": lane_last,
                    "physical_cores": lane_workers,
                    "available_cores": lane_workers,
                    "worker_process_limit": max_workers,
                    "worker_lane_index": lane_idx,
                    "worker_lane_count": lanes,
                    "unsplit_workers": workers,
                    "unsplit_items": items,
                    "unsplit_start": start,
                    "unsplit_end": end,
                })
                out.append(lane)
        else:
            out.append(base)
    for shard_idx, p in enumerate(out):
        p["shard_index"] = shard_idx
        p["num_shards"] = len(out)
    return out


def _cpu_batch_item_counts(args) -> tuple:
    """Return (logical_items, item_multiplier, total_work_items) for CPU batch CLIs."""
    try:
        logical_items = int(getattr(args, "items", 0) or 0)
        item_multiplier = int(getattr(args, "item_multiplier", 1) or 1)
    except Exception:
        sys.exit("--items and --item-multiplier must be integers")
    if logical_items < 0:
        sys.exit("--items must be >= 0")
    if item_multiplier < 1:
        sys.exit("--item-multiplier must be >= 1")
    return logical_items, item_multiplier, logical_items * item_multiplier


def _cpu_wave_summary(items: int, workers: int) -> dict:
    """Compact per-wave item counts for CPU batch audit logs."""
    items = max(0, int(items or 0))
    workers = max(0, int(workers or 0))
    if items <= 0 or workers <= 0:
        return {"waves": 0, "workers": workers, "last_wave_items": 0, "wave_items": []}
    waves = _ceil_div(items, workers)
    last = items - workers * max(0, waves - 1)
    if last <= 0:
        last = workers
    if waves <= 32:
        wave_items = [min(workers, max(0, items - i * workers)) for i in range(waves)]
        return {
            "waves": waves,
            "workers": workers,
            "last_wave_items": last,
            "wave_items": wave_items,
        }
    head = [min(workers, max(0, items - i * workers)) for i in range(8)]
    tail_start = max(8, waves - 4)
    tail = [min(workers, max(0, items - i * workers)) for i in range(tail_start, waves)]
    return {
        "waves": waves,
        "workers": workers,
        "last_wave_items": last,
        "wave_items_head": head,
        "wave_items_tail": tail,
        "wave_items_omitted": max(0, waves - len(head) - len(tail)),
    }


def _cpu_batch_log_payload(total_items: int, plan: list, node_states: Optional[dict] = None,
                           use_total_cores: bool = False, templates: Optional[dict] = None,
                           logical_items: Optional[int] = None,
                           item_multiplier: int = 1) -> dict:
    """Detailed JSONL payload for CPU batch planning and submission."""
    node_states = node_states or {}
    nodes = []
    for p in plan or []:
        node = p.get("node")
        st = node_states.get(node) or {}
        entry = {
            "node": node,
            "range": [int(p.get("start") or 0), int(p.get("end") or 0)],
            "assigned_items": int(p.get("items") or 0),
            "free_physical_cores_used_for_plan": int(p.get("physical_cores") or 0),
            "available_physical_cores": int(p.get("available_cores") or p.get("physical_cores") or 0),
            "total_physical_cores": int(p.get("total_physical_cores") or p.get("physical_cores") or 0),
            "workers": int(p.get("workers") or 0),
            "waves": int(p.get("waves") or 0),
            "global_waves": int(p.get("global_waves") or 0),
            "last_wave_items": int(p.get("last_wave_items") or 0),
            "shard_index": int(p.get("shard_index") or 0),
            "num_shards": int(p.get("num_shards") or 1),
            "wave_plan": _cpu_wave_summary(p.get("items") or 0, p.get("workers") or 0),
            "live_node": {
                "alive": st.get("alive"),
                "free_cpu": st.get("free_cpu"),
                "total_cpu": st.get("total_cpu"),
                "logical_cpu": st.get("logical_cpu") or st.get("logical_cores"),
                "free_ram_mb": st.get("free_ram_mb"),
                "total_ram_mb": st.get("total_ram_mb"),
                "running_count": st.get("running_count"),
                "os": st.get("os"),
            },
        }
        nodes.append(entry)
    total_free = sum(int(p.get("physical_cores") or 0) for p in plan or [])
    total_capacity = sum(int(p.get("total_physical_cores") or p.get("physical_cores") or 0)
                         for p in plan or [])
    return {
        "total_items": int(total_items or 0),
        "logical_items": int(logical_items if logical_items is not None else total_items or 0),
        "item_multiplier": int(item_multiplier or 1),
        "node_count": len(plan or []),
        "free_physical_cores_used_for_plan": total_free,
        "total_physical_cores": total_capacity,
        "global_waves": max([int(p.get("global_waves") or p.get("waves") or 0)
                             for p in plan or []] or [0]),
        "use_total_cores": bool(use_total_cores),
        "nodes": nodes,
        "templates": templates or {},
    }


def _cpu_parallel_template_values(plan: dict, total_items: Optional[int] = None,
                                  logical_items: Optional[int] = None,
                                  item_multiplier: int = 1) -> dict:
    total = total_items if total_items is not None else plan.get("items", 0)
    logical = logical_items if logical_items is not None else total
    multiplier = item_multiplier or 1
    vals = {
        "node": plan.get("node", ""),
        "start": plan.get("start", 0),
        "end": plan.get("end", plan.get("items", 0)),
        "items": plan.get("items", 0),
        "shard_items": plan.get("items", 0),
        "total_items": total,
        "total_work_items": total,
        "logical_items": logical,
        "base_items": logical,
        "item_multiplier": multiplier,
        "items_per_unit": multiplier,
        "episodes_per_item": multiplier,
        "workers": plan.get("workers", 0),
        "n_workers": plan.get("workers", 0),
        "num_workers": plan.get("workers", 0),
        "waves": plan.get("waves", 0),
        "rounds": plan.get("waves", 0),
        "physical_cores": plan.get("physical_cores", 0),
        "available_cores": plan.get("available_cores", plan.get("physical_cores", 0)),
        "total_physical_cores": plan.get("total_physical_cores", plan.get("physical_cores", 0)),
        "last_wave_items": plan.get("last_wave_items", 0),
        "shard_index": plan.get("shard_index", 0),
        "shard_id": plan.get("shard_index", 0),
        "num_shards": plan.get("num_shards", 1),
    }
    return {k: str(v) for k, v in vals.items()}


def _format_cpu_parallel_template(text: Optional[str], plan: dict,
                                  total_items: Optional[int] = None,
                                  logical_items: Optional[int] = None,
                                  item_multiplier: int = 1) -> Optional[str]:
    if text is None:
        return None
    out = str(text)
    for key, val in _cpu_parallel_template_values(
            plan, total_items, logical_items, item_multiplier).items():
        out = out.replace("{" + key + "}", val)
    return out


_CPU_WORKER_FLAG_RE = re.compile(
    r"(?P<flag>--(?:workers|n-workers|n_workers|num-workers|num_workers|jobs|n-jobs|n_jobs|num-jobs|num_jobs))"
    r"(?P<sep>=|\s+)"
    r"(?P<value>auto|AUTO|\{workers\}|\{n_workers\}|\{num_workers\})"
)


def _rewrite_cpu_parallel_cmd(cmd: str, plan: dict, total_items: Optional[int] = None,
                              logical_items: Optional[int] = None,
                              item_multiplier: int = 1) -> str:
    out = _format_cpu_parallel_template(
        cmd or "", plan, total_items, logical_items, item_multiplier) or ""
    workers = str(plan.get("workers", 0))

    def repl(m):
        return f"{m.group('flag')}{m.group('sep')}{workers}"

    return _CPU_WORKER_FLAG_RE.sub(repl, out)


def _cpu_parallel_env(task: dict) -> dict:
    keys = {
        "SCHEDULEURM_CPU_TOTAL_ITEMS": task.get("cpu_parallel_total_items")
                                      or task.get("cpu_parallel_items"),
        "SCHEDULEURM_CPU_TOTAL_WORK_ITEMS": task.get("cpu_parallel_total_items")
                                             or task.get("cpu_parallel_items"),
        "SCHEDULEURM_CPU_LOGICAL_ITEMS": task.get("cpu_parallel_logical_items"),
        "SCHEDULEURM_CPU_ITEM_MULTIPLIER": task.get("cpu_parallel_item_multiplier"),
        "SCHEDULEURM_CPU_ITEMS_PER_UNIT": task.get("cpu_parallel_item_multiplier"),
        "SCHEDULEURM_CPU_EPISODES_PER_ITEM": task.get("cpu_parallel_item_multiplier"),
        "SCHEDULEURM_CPU_SHARD_START": task.get("cpu_parallel_start"),
        "SCHEDULEURM_CPU_SHARD_END": task.get("cpu_parallel_end"),
        "SCHEDULEURM_CPU_SHARD_ITEMS": task.get("cpu_parallel_items"),
        "SCHEDULEURM_CPU_WORKERS": task.get("cpu_auto_workers"),
        "SCHEDULEURM_CPU_WAVES": task.get("cpu_parallel_waves"),
        "SCHEDULEURM_CPU_PHYSICAL_CORES": task.get("cpu_parallel_physical_cores"),
        "SCHEDULEURM_CPU_TOTAL_PHYSICAL_CORES": task.get("cpu_parallel_total_physical_cores"),
        "SCHEDULEURM_CPU_LAST_WAVE_ITEMS": task.get("cpu_parallel_last_wave_items"),
        "SCHEDULEURM_CPU_SHARD_INDEX": task.get("cpu_parallel_shard_index"),
        "SCHEDULEURM_CPU_NUM_SHARDS": task.get("cpu_parallel_num_shards"),
    }
    return {k: str(v) for k, v in keys.items() if v is not None and v != ""}


def _apply_cpu_parallel_plan_to_task(task: dict, node_state: Optional[dict] = None) -> Optional[dict]:
    items = int(task.get("cpu_parallel_items") or 0)
    if items <= 0:
        return None
    node = task.get("node")
    batch_plan = task.get("cpu_batch_plan") if isinstance(task.get("cpu_batch_plan"), dict) else {}
    if batch_plan:
        physical = int(batch_plan.get("physical_cores") or batch_plan.get("available_cores")
                       or _node_physical_cores(node, node_state))
        plan = _cpu_worker_plan_for_items(items, physical)
        workers = int(batch_plan.get("workers") or task.get("cpu_auto_workers") or plan["workers"])
        waves = int(batch_plan.get("waves") or _ceil_div(items, max(1, workers)))
        last = int(batch_plan.get("last_wave_items") or (items - workers * max(0, waves - 1)))
        if last <= 0:
            last = workers
        plan.update({
            "workers": workers,
            "waves": waves,
            "last_wave_items": last,
            "physical_cores": physical,
            "total_physical_cores": int(batch_plan.get("total_physical_cores") or physical),
        })
    else:
        physical = _node_physical_cores(node, node_state)
        plan = _cpu_worker_plan_for_items(items, physical)
    plan.update({
        "node": node or "",
        "start": int(task.get("cpu_parallel_start") or 0),
        "end": int(task.get("cpu_parallel_end") or items),
        "shard_index": int(task.get("cpu_parallel_shard_index") or 0),
        "num_shards": int(task.get("cpu_parallel_num_shards") or 1),
    })
    task["cpu_auto_workers"] = int(plan["workers"])
    task["cpu_parallel_waves"] = int(plan["waves"])
    task["cpu_parallel_physical_cores"] = int(plan["physical_cores"])
    task["cpu_parallel_total_physical_cores"] = int(plan.get("total_physical_cores") or plan["physical_cores"])
    task["cpu_parallel_last_wave_items"] = int(plan["last_wave_items"])
    if int(task.get("cpu_cores") or 0) < int(plan["workers"]):
        task["cpu_cores"] = int(plan["workers"])
    old_cmd = task.get("cmd") or ""
    new_cmd = _rewrite_cpu_parallel_cmd(
        old_cmd, plan, task.get("cpu_parallel_total_items") or items,
        task.get("cpu_parallel_logical_items"),
        int(task.get("cpu_parallel_item_multiplier") or 1))
    if new_cmd != old_cmd:
        task["cmd"] = new_cmd
        task["cpu_auto_worker_cmd_rewritten"] = True
    return plan


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


def _mark_probe_unknown(task: dict, res: Optional[dict] = None) -> None:
    """Remember that a running task's node/backend could not be probed.

    Unknown is deliberately not terminal: remote SSH/Slurm outages must not
    turn into false done/failed or duplicate relaunches. These markers make the
    blind window visible and let the next successful probe record a reconnect
    sync audit trail.
    """
    now = time.time()
    if not task.get("probe_unknown_since"):
        task["probe_unknown_since"] = now
    task["last_probe_unknown_at"] = now
    task["probe_unknown_count"] = int(task.get("probe_unknown_count") or 0) + 1
    err = (res or {}).get("error") or (res or {}).get("terminal_reason")
    task["last_probe_unknown_reason"] = str(err or "backend probe returned unknown")[:300]


def _clear_probe_unknown(task: dict, now: Optional[float] = None) -> Optional[dict]:
    """Clear current unknown-probe markers and return a sync record if any."""
    since = task.get("probe_unknown_since")
    if not since:
        return None
    now = now or time.time()
    try:
        dur = max(0, int(now - float(since)))
    except Exception:
        dur = 0
    rec = {
        "synced_at": now,
        "duration_s": dur,
        "count": int(task.get("probe_unknown_count") or 0),
        "reason": task.get("last_probe_unknown_reason") or "backend probe returned unknown",
    }
    task["last_reconnected_at"] = now
    task["last_probe_unknown_duration_s"] = dur
    task["last_probe_unknown_count"] = rec["count"]
    task["last_probe_unknown_reason"] = rec["reason"]
    task.pop("probe_unknown_since", None)
    task.pop("last_probe_unknown_at", None)
    task.pop("probe_unknown_count", None)
    return rec


def _annotate_diag_after_unknown(diag: dict, sync_rec: Optional[dict]) -> dict:
    """Attach reconnect-sync context to a terminal diagnosis."""
    if not sync_rec:
        return diag
    diag = dict(diag or {})
    dur = int(sync_rec.get("duration_s") or 0)
    count = int(sync_rec.get("count") or 0)
    note = f"terminal state synced after remote probe was unknown for {dur}s ({count} probe cycle(s))"
    reason = diag.get("reason") or ""
    diag["reason"] = f"{reason}; {note}" if reason else note
    diag["offline_sync_s"] = dur
    diag["offline_probe_unknown_count"] = count
    diag["offline_probe_unknown_reason"] = sync_rec.get("reason")
    return diag

_LIVE_REMAINING_ETA_SOURCES = {
    "tqdm",
    "inline_eta",
    "progress_rate",
    "bapr_seed_batch",
    "runtime_history_fallback",
    "duration_ewma_fallback",
}
_LIVE_RUNTIME_PROJECTION_SOURCES = {"tqdm", "inline_eta", "progress_rate", "bapr_seed_batch"}
_ETA_AUDIT_FIELDS = (
    "eta_seconds", "eta_source", "eta_confidence", "eta_updated_at",
    "eta_detail", "eta_log_bytes", "eta_probe_error", "last_progress_line",
)
_RUNTIME_PROJECTION_FIELDS = (
    "runtime_total_s_est", "runtime_eta_s_est", "runtime_est_source",
    "runtime_progress_at", "runtime_current_unit", "runtime_total_units",
    "runtime_unit_s_est",
)

def _eta_source_base(source: object) -> str:
    return str(source or "").split(":", 1)[0]

def _is_live_remaining_eta_source(source: object) -> bool:
    return _eta_source_base(source) in _LIVE_REMAINING_ETA_SOURCES

def _is_live_runtime_projection_source(source: object) -> bool:
    return _eta_source_base(source) in _LIVE_RUNTIME_PROJECTION_SOURCES

def _clear_live_eta_fields(task: dict, clear_runtime_projection: bool = False) -> bool:
    """Remove running-run ETA/progress fields before a task re-enters queue."""
    changed = False
    for k in _ETA_AUDIT_FIELDS:
        if k in task:
            task.pop(k, None)
            changed = True
    if clear_runtime_projection or _is_live_runtime_projection_source(task.get("runtime_est_source")):
        for k in _RUNTIME_PROJECTION_FIELDS:
            if k in task:
                task.pop(k, None)
                changed = True
    return changed

def _queued_has_stale_live_eta(task: dict) -> bool:
    if task.get("status") not in ("queued", "launching"):
        return False
    if _is_live_remaining_eta_source(task.get("eta_source")):
        return True
    if _is_live_runtime_projection_source(task.get("runtime_est_source")):
        return True
    if task.get("last_progress_line") and not task.get("started_at"):
        return True
    return False

def _eta_confidence_for_source(source: str) -> str:
    base = _eta_source_base(source)
    if base in ("tqdm", "inline_eta", "local_test_tqdm"):
        return "high"
    if base in ("progress_rate", "local_test_progress", "runtime_history", "bapr_seed_batch"):
        return "medium"
    return "low"

def _fmt_eta_seconds(seconds: int) -> str:
    seconds = int(seconds or 0)
    if seconds <= 0:
        return "?"
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"

def _eta_source_tag(source: object) -> str:
    base = _eta_source_base(source)
    if base == "tqdm":
        return "tqdm"
    if base == "inline_eta":
        return "logeta"
    if base == "progress_rate":
        return "prog"
    if base == "bapr_seed_batch":
        return "bapr"
    if base in ("runtime_history", "runtime_history_fallback"):
        return "hist"
    if base in ("duration_ewma", "duration_ewma_fallback"):
        return "ewma"
    if base.startswith("local_test"):
        return "test"
    if base.startswith("closest"):
        return "near"
    return base or "unknown"

def _format_task_eta(task: dict) -> str:
    eta = int(task.get("eta_seconds") or 0)
    if eta <= 0:
        return ""
    return f"eta~{_fmt_eta_seconds(eta)}/{_eta_source_tag(task.get('eta_source'))}"

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
        elif node and _node_is_windows(node):
            return _fetch_windows_text_tail(task, log_path, max_bytes=4096)
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
        elif node and _node_is_windows(node):
            return _scan_windows_log_for_patterns(task, SUCCESS_PATTERNS)
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
            elif node and _node_is_windows(node):
                tail_text, log_size = _fetch_windows_text_tail(task, log_path, max_bytes=4096)
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
                elif node and _node_is_windows(node):
                    tail_text, log_size = _fetch_windows_text_tail(task, real_log_path, max_bytes=4096)
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
    if not success_matched:
        final_model_success = _terminal_final_model_success(task)
        if final_model_success:
            success_matched = [final_model_success]
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
            if not _node_is_windows(task.get("node") or ""):
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
    launch_state, launch_reason = _recorded_launch_safety_state(parent)
    if launch_state in ("alive", "unknown"):
        parent["last_block_reason"] = (
            f"not auto-requeued: recorded launch artifact is {launch_state} "
            f"({launch_reason}); refusing to duplicate the same run"
        )
        return None
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
        "origin": "scheduleurm",
        "submitted_by": _local_user(),
        "submitted_host": _local_host_short(),
        "scheduler_id": _ClaimManager.scheduler_id(),
        "process_group": None,
        "_diagnosis": None,
        "result_artifacts": [],
        "result_artifacts_discovered_at": None,
        "notified_launch": False,
        "notified_done": False,
        "retry_count": retry_n,
        "parent_id": parent["id"],
        "last_block_reason": f"auto-requeue (retry {retry_n}/{MAX_AUTO_RETRY}) after {parent['id']} crashed",
    })
    _clear_live_eta_fields(new_task, clear_runtime_projection=True)
    for k in ("requeued_as", "cancelled_at", "cancelled_by_user", "cancel_reason",
              "cancelled_by", "cancel_actor",
              "last_killed_at", "last_killed_by", "last_kill_actor",
              "last_kill_action", "last_kill_reason",
              "claim_intent_node", "claim_intent_nodes", "claim_intent_at",
              "probe_unknown_since", "last_probe_unknown_at", "probe_unknown_count",
              "last_probe_unknown_duration_s", "last_probe_unknown_count",
              "last_probe_unknown_reason", "last_reconnected_at",
              "cpu_fallback_selected", "cpu_fallback_original_vram_mb",
              "cpu_fallback_capability", "windows_pin_base", "windows_pin_cores",
              "bootstrap_stdout_path", "bootstrap_stderr_path", "local_ssh_log_path",
              "wrapper_pid_path", "last_launch_pre_snapshot",
              "last_launch_post_snapshot", "last_node", "last_gpu_idx",
              # Retry clones must re-scan checkpoints and re-decide placement.
              # A stale migration pin or resume scan can otherwise hard-pin the
              # retry to an old staged node or miss a newer checkpoint written
              # just before the crash.
              "staged_node", "migrated_from", "migrated_at",
              "resume_locations", "resume_scan_errors", "resume_preferred_nodes",
              "resume_checkpoint_node", "resume_scan_at", "resume_scan_key"):
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
            # ssh failed for this node OR task wasn't probed. Default is
            # conservative (avoid duplicate launch). CPU batch shards can opt
            # into reroute because their item ranges are independent and
            # result writes should be idempotent/resumable.
            _mark_probe_unknown(t, res)
            if t.get("reroute_on_node_down"):
                try:
                    unknown_for = time.time() - float(t.get("probe_unknown_since") or time.time())
                except Exception:
                    unknown_for = 0
                threshold = int(t.get("node_down_requeue_s") or NODE_DOWN_REQUEUE_S)
                if unknown_for >= threshold:
                    launch_state, launch_reason = _recorded_launch_safety_state(t)
                    if launch_state in ("alive", "unknown"):
                        t["last_block_reason"] = (
                            f"node/probe unknown for {int(unknown_for)}s on {t.get('node')}; "
                            f"not rerouting because recorded launch artifact is {launch_state} "
                            f"({launch_reason})"
                        )
                        continue
                    reason = (
                        f"node/probe unknown for {int(unknown_for)}s on {t.get('node')}; "
                        "rerouting opt-in CPU batch shard"
                    )
                    _set_current_usage(t, 0, 0, 0.0)
                    t["status"] = "failed"
                    t["finished_at"] = time.time()
                    t["last_block_reason"] = reason
                    t["_diagnosis"] = {
                        "is_crash": True,
                        "reason": reason,
                        "tail": t.get("last_probe_unknown_reason") or "backend probe unknown",
                        "lifetime_s": int(max(0, t["finished_at"] - (t.get("started_at") or t["finished_at"]))),
                        "log_path": t.get("log_path"),
                    }
                    try:
                        _release_task_claims_and_intents(t)
                    except Exception:
                        pass
                    new_id = _requeue_after_crash(t, state)
                    if new_id:
                        t["requeued_as"] = new_id
            continue
        reconnect_sync = _clear_probe_unknown(t)
        if res["state"] == "dead":
            if _local_launch_transport_alive(t):
                _mark_probe_unknown(
                    t,
                    {"error": "backend reported dead but local launch transport is still alive"},
                )
                continue
            _set_current_usage(t, 0, 0, 0.0)
            t["status"] = "done"
            t["finished_at"] = time.time()
            if res.get("exit_code") is not None:
                t["exit_code"] = res.get("exit_code")
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
            diag = _annotate_diag_after_unknown(diag, reconnect_sync)
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
            # Best-effort result indexing for completed/terminal tasks. This is
            # deliberately separate from --result-dir sync: local tasks and older
            # submissions may still have useful output paths in cmd flags or logs.
            try:
                _record_result_artifacts(t)
            except Exception:
                pass
            duration_s = 0
            if t.get("started_at") and t.get("finished_at"):
                duration_s = max(0, int(t["finished_at"] - t["started_at"]))
            peak_vram_for_history = t.get("peak_vram_mb", 0)
            peak_ram_for_history = t.get("peak_ram_mb", 0)
            duration_for_history = duration_s if t.get("status") == "done" else 0
            if _untrusted_startup_oom_sample(t, duration_s):
                t["history_skipped_reason"] = (
                    "startup OOM before progress; peak likely JAX/CUDA preallocation, "
                    "not steady-state resource need"
                )
                peak_vram_for_history = 0
                peak_ram_for_history = 0
            history_record(
                t.get("signature"),
                peak_vram_mb=peak_vram_for_history,
                peak_ram_mb=peak_ram_for_history,
                cpu_cores=t.get("cpu_cores", 0),
                duration_s=duration_for_history,
            )
            if t.get("status") == "done":
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
    now = time.time()
    last_probe = float(state.get("_last_running_probe_at") or 0)
    if RUNNING_PROBE_MIN_INTERVAL_S <= 0 or now - last_probe >= RUNNING_PROBE_MIN_INTERVAL_S:
        _batch_check_running(state)
        state["_last_running_probe_at"] = time.time()
    last_eta = float(state.get("_last_eta_refresh_at") or 0)
    if ETA_REFRESH_MIN_INTERVAL_S <= 0 or now - last_eta >= ETA_REFRESH_MIN_INTERVAL_S:
        _refresh_eta_from_logs(state)
        state["_last_eta_refresh_at"] = time.time()


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
        runtime_total = _runtime_total_history_s(t)
        ewma = runtime_total or int(h.get("dur_s_ewma", 0))
        elapsed = _effective_elapsed_s(t)
        t["eta_seconds"] = eta_tracker.compute_eta_seconds(
            "", elapsed_s=elapsed, fallback_ewma_s=ewma, cmd=t.get("cmd"),
        )
        t["eta_updated_at"] = int(time.time())
        t["eta_log_bytes"] = 0
        t["eta_detail"] = "no scheduler log_path; ETA from runtime history fallback" if runtime_total else "no scheduler log_path; ETA from duration EWMA fallback"
        t.pop("eta_probe_error", None)
        if t.get("eta_seconds"):
            t["eta_source"] = "runtime_history_fallback" if runtime_total else "duration_ewma_fallback"
            t["eta_confidence"] = "low"

    if not by_node:
        return

    import re as _re

    def _probe(node):
        entries = by_node[node]
        if _node_is_windows(node):
            parts = []
            for (t, log_path) in entries:
                tid = t["id"]
                parts.append(f"Write-Output {_ps_quote(f'===ETA_LOG_{tid}===')}")
                parts.append(_windows_tail_ps(_windows_path_for_task(t, log_path), max_bytes=4096))
            try:
                rc, out, _ = _run_windows_ps(node, "\n".join(parts), timeout=30, check=False)
                return out if rc == 0 else None
            except Exception:
                return None
        # Build a single ssh cmd that tails each task's log with a marker separator.
        # `tail -c 4096` returns at most 4KB which is plenty for the tqdm/epoch
        # patterns we need. `2>/dev/null` swallows missing-log errors. The trailing
        # `; true` keeps overall rc=0 even if some tails fail.
        parts = []
        for (t, log_path) in entries:
            tid = t["id"]
            q_log = shlex.quote(log_path)
            parts.append(f"echo '===ETA_LOG_{tid}==='")
            parts.append(
                f"(tail -c 4096 {q_log} 2>/dev/null; "
                f"grep -a 'Results saved to:' {q_log} 2>/dev/null | tail -n 100)"
            )
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
                t["eta_probe_error"] = "log tail probe failed; kept previous ETA or used fallback"
                t["eta_updated_at"] = int(time.time())
                if t.get("eta_seconds") is None:
                    sig = t.get("signature") or ""
                    h = history_get(sig) or {}
                    runtime_total = _runtime_total_history_s(t)
                    ewma = runtime_total or int(h.get("dur_s_ewma", 0))
                    elapsed = _effective_elapsed_s(t)
                    t["eta_seconds"] = eta_tracker.compute_eta_seconds(
                        "", elapsed_s=elapsed, fallback_ewma_s=ewma, cmd=t.get("cmd"),
                    )
                    if t.get("eta_seconds"):
                        t["eta_source"] = "runtime_history_fallback" if runtime_total else "duration_ewma_fallback"
                        t["eta_confidence"] = "low"
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
            runtime_total = _runtime_total_history_s(t)
            ewma = runtime_total or int(h.get("dur_s_ewma", 0))
            elapsed = _effective_elapsed_s(t)
            t["eta_seconds"] = eta_tracker.compute_eta_seconds(
                tail_text, elapsed_s=elapsed, fallback_ewma_s=ewma, cmd=t.get("cmd"),
            )
            t["eta_updated_at"] = int(time.time())
            t["eta_log_bytes"] = len(tail_text.encode("utf-8", errors="replace"))
            t.pop("eta_probe_error", None)
            progress_line = _last_progress_line(tail_text)
            if progress_line:
                t["last_progress_line"] = progress_line
            projection = eta_tracker.runtime_projection(
                tail_text, elapsed_s=elapsed, cmd=t.get("cmd"))
            bapr_projection = _bapr_batch_projection(t, tail_text, elapsed)
            if bapr_projection:
                projection = bapr_projection
                t["eta_seconds"] = int(bapr_projection.get("eta_s") or 0)
            _apply_runtime_projection(t, projection)
            if projection:
                source = projection.get("source") or "progress"
                t["eta_source"] = source
                t["eta_confidence"] = _eta_confidence_for_source(source)
                if source == "bapr_seed_batch":
                    done = projection.get("completed_seeds")
                    total = projection.get("seed_count")
                    suffix = f"; completed_seeds={done}/{total}" if done is not None and total is not None else ""
                    t["eta_detail"] = f"parsed BAPR seed batch progress from scheduler log tail{suffix}"
                else:
                    t["eta_detail"] = f"parsed {source} from scheduler log tail"
            elif t.get("eta_seconds"):
                t["eta_source"] = "runtime_history_fallback" if runtime_total else "duration_ewma_fallback"
                t["eta_confidence"] = "low"
                t["eta_detail"] = "no progress marker in log tail; ETA from runtime history fallback" if runtime_total else "no progress marker in log tail; ETA from duration EWMA fallback"
            else:
                t["eta_detail"] = "no parseable progress marker and no runtime history fallback"

# ---------- placement + resource forensics ----------
def _node_ram_headroom_mb(node_state: dict, node_info: dict) -> int:
    fixed_headroom = node_info.get("ram_headroom_mb")
    if fixed_headroom is not None:
        return max(0, int(fixed_headroom))
    frac = node_info.get("ram_headroom_frac", RAM_HEADROOM_FRAC)
    # Use the probed total when available so headroom tracks reality, not config; the config
    # value is only used as an upper bound (already enforced in probe_node via min()).
    total_for_headroom = node_state.get("total_ram_mb") or node_info.get("ram_mb", 0)
    return int(total_for_headroom * frac)


def _gpu_freeze_line_mb(total_mb: int) -> int:
    if total_mb <= 0:
        return 0
    return min(total_mb, total_mb // 3 + max(0, int(ONE_THIRD_PACK_GRACE_MB)))


def _gpu_threshold_snapshot(gpu: Optional[dict]) -> Optional[dict]:
    if not gpu:
        return None
    total = int(gpu.get("total_mb") or 0)
    used = int(gpu.get("used_mb") or 0)
    free = int(gpu.get("free_mb") or max(0, total - used))
    third = total // 3 if total > 0 else 0
    freeze = _gpu_freeze_line_mb(total)
    return {
        "idx": int(gpu.get("idx") or 0),
        "used_mb": used,
        "free_mb": free,
        "total_mb": total,
        "util_pct": int(gpu.get("util_pct") or 0),
        "one_third_mb": third,
        "freeze_line_mb": freeze,
        "over_one_third": bool(total > 0 and used >= third and used > 100),
        "over_freeze_line": bool(total > 0 and used >= freeze and used > 100),
        "over_full": bool(total > 0 and used >= total),
        "used_pct": int(round(100 * used / max(total, 1))),
    }


def _node_ram_snapshot(node_state: Optional[dict]) -> Optional[dict]:
    if not node_state:
        return None
    node_info = NODES.get(node_state.get("name"), {})
    total = int(node_state.get("total_ram_mb") or node_info.get("ram_mb") or 0)
    free = int(node_state.get("free_ram_mb") or 0)
    headroom = _node_ram_headroom_mb(node_state, node_info)
    grace = max(0, int(RAM_HEADROOM_EVICTION_GRACE_MB))
    eviction_headroom = max(0, headroom - grace)
    return {
        "free_mb": free,
        "total_mb": total,
        "headroom_mb": headroom,
        "eviction_headroom_mb": eviction_headroom,
        "headroom_grace_mb": grace,
        "below_headroom": bool(free < headroom),
        "below_eviction_headroom": bool(free < eviction_headroom),
        "available_after_headroom_mb": free - headroom,
        "available_after_eviction_headroom_mb": free - eviction_headroom,
        "headroom_gap_mb": max(0, headroom - free),
        "eviction_gap_mb": max(0, eviction_headroom - free),
    }


def _task_progress_ratio(task: dict) -> Optional[float]:
    total = int(task.get("runtime_total_units") or 0)
    current = int(task.get("runtime_current_unit") or 0)
    if total <= 0 or current < 0:
        return None
    return max(0.0, min(1.0, float(current) / float(total)))


def _task_has_progress_evidence(task: dict) -> bool:
    ratio = _task_progress_ratio(task)
    if ratio is not None and ratio > 0.0:
        return True
    if int(task.get("runtime_current_unit") or 0) > 0:
        return True
    line = task.get("last_progress_line") or ""
    return bool(re.search(r"\b(?:Iter|Epoch|Step|Episode)\s*[#:]?\s*[1-9]\d*\b", line))


def _task_ram_pressure_mb(task: dict) -> int:
    return max(
        int(task.get("current_ram_mb") or 0),
        int(task.get("peak_ram_mb") or 0),
        int(task.get("ram_mb") or 0),
    )


def _task_has_resume_evidence(task: dict) -> bool:
    return bool(task.get("resume_locations") or task.get("resume_from") or task.get("resume_checkpoint_node"))


def _ckpt_task_evict_protected(task: dict) -> bool:
    """Avoid killing meaningful checkpoint work unless resume evidence exists."""
    if not task.get("ckpt_dir"):
        return False
    if _task_has_resume_evidence(task):
        return False
    elapsed = _effective_elapsed_s(task)
    progress = _task_progress_ratio(task)
    has_progress = progress is not None and progress >= float(RAM_EVICT_CKPT_PROTECT_PROGRESS)
    old_enough = elapsed >= float(RAM_EVICT_CKPT_PROTECT_MIN_AGE_S)
    return bool(old_enough or has_progress)


def _task_has_incremental_result_resume(task: dict) -> bool:
    """Detect eval-style resumability via incremental result artifacts.

    offline-sumo eval jobs do not have a training ckpt_dir, but commands like
    ``--skip_existing --out_csv results.csv`` resume from the result CSV. Treat
    these as meaningful in-progress work for eviction/preemption decisions.
    """
    cmd = task.get("cmd") or ""
    if "--skip_existing" not in cmd and "--skip-existing" not in cmd:
        return False
    artifacts = _discover_result_artifacts(task, include_log=False)
    return any((rec.get("kind") == "file" and rec.get("path")) for rec in artifacts)


def _task_evict_loss_protected(task: dict) -> bool:
    return bool(_ckpt_task_evict_protected(task) or _task_has_incremental_result_resume(task))


def _summarize_task_for_resource_log(task: dict) -> dict:
    started = task.get("started_at")
    elapsed = max(0, int(time.time() - started)) if started else 0
    ratio = _task_progress_ratio(task)
    resume_locations = task.get("resume_locations") or []
    out = {
        "id": task.get("id"),
        "status": task.get("status"),
        "project": task.get("project"),
        "signature": task.get("signature"),
        "node": task.get("node"),
        "gpu_idx": task.get("gpu_idx"),
        "started_at": started,
        "elapsed_s": elapsed,
        "eta_seconds": int(task.get("eta_seconds") or 0),
        "eta_source": task.get("eta_source") or task.get("runtime_est_source"),
        "runtime_current_unit": task.get("runtime_current_unit"),
        "runtime_total_units": task.get("runtime_total_units"),
        "progress_ratio": ratio,
        "current_vram_mb": int(task.get("current_vram_mb") or 0),
        "peak_vram_mb": int(task.get("peak_vram_mb") or 0),
        "current_ram_mb": int(task.get("current_ram_mb") or 0),
        "peak_ram_mb": int(task.get("peak_ram_mb") or 0),
        "cpu_cores": int(task.get("cpu_cores") or 0),
        "windows_pin_base": task.get("windows_pin_base"),
        "windows_pin_cores": task.get("windows_pin_cores"),
        "ram_pressure_mb": _task_ram_pressure_mb(task),
        "ckpt_dir": task.get("ckpt_dir"),
        "resume_scan_at": task.get("resume_scan_at"),
        "resume_locations_count": len(resume_locations),
        "resume_checkpoint_node": task.get("resume_checkpoint_node"),
        "ckpt_evict_protected": _ckpt_task_evict_protected(task),
        "incremental_result_resume": _task_has_incremental_result_resume(task),
        "evict_loss_protected": _task_evict_loss_protected(task),
        "last_progress_line": (task.get("last_progress_line") or "")[-240:],
    }
    return out


def _assign_windows_pin_plan(task: dict, state: dict, node_state: Optional[dict]):
    node = task.get("node")
    if not node or not _node_is_windows(node):
        task.pop("windows_pin_base", None)
        task.pop("windows_pin_cores", None)
        return
    total = max(1, int(_node_physical_cores(node, node_state) or NODES.get(node, {}).get("cpu_cores") or 1))
    width = max(1, min(total, int(task.get("cpu_cores") or DEFAULT_CPU_CORES or 1)))
    occupied = [False] * total
    legacy_cursor = 0
    running = [
        t for t in state.get("tasks", [])
        if t is not task
        and t.get("status") in ("running", "launching")
        and (t.get("node") or t.get("assigned_node")) == node
    ]
    running.sort(key=lambda t: t.get("started_at") or t.get("submitted_at") or 0)
    for other in running:
        other_width = max(1, min(total, int(other.get("windows_pin_cores")
                                            or other.get("cpu_cores")
                                            or DEFAULT_CPU_CORES
                                            or 1)))
        base = other.get("windows_pin_base")
        if base is None:
            base = legacy_cursor
            legacy_cursor += other_width
        try:
            base = int(base)
        except Exception:
            continue
        for i in range(other_width):
            occupied[(base + i) % total] = True

    def fits(start: int) -> bool:
        return all(not occupied[(start + i) % total] for i in range(width))

    base = None
    for start in range(total):
        if fits(start):
            base = start
            break
    if base is None:
        base = min(total - 1, legacy_cursor % total)
    task["windows_pin_base"] = int(base)
    task["windows_pin_cores"] = int(width)


def _resource_snapshot_for_placement(state: dict, nodes: list, task: dict,
                                     node_name: Optional[str], gpu_idx,
                                     stage: str) -> dict:
    node_state = next((n for n in nodes if n.get("name") == node_name), None)
    gpu_state = None
    if node_state and gpu_idx is not None:
        for g in node_state.get("gpus") or []:
            if g.get("idx") == gpu_idx:
                gpu_state = g
                break
    same_gpu_tasks = []
    same_node_tasks = []
    for t in state.get("tasks", []):
        if t.get("status") not in ("running", "launching"):
            continue
        if node_name and t.get("node") == node_name:
            same_node_tasks.append(_summarize_task_for_resource_log(t))
            if gpu_idx is not None and t.get("gpu_idx") == gpu_idx:
                same_gpu_tasks.append(_summarize_task_for_resource_log(t))
    same_gpu_tasks.sort(key=lambda x: x.get("started_at") or 0)
    same_node_tasks.sort(key=lambda x: x.get("started_at") or 0)
    return {
        "stage": stage,
        "ts": time.time(),
        "task": _summarize_task_for_resource_log(task),
        "target_node": node_name,
        "target_gpu_idx": gpu_idx,
        "gpu": _gpu_threshold_snapshot(gpu_state),
        "node_ram": _node_ram_snapshot(node_state),
        "same_gpu_tasks": same_gpu_tasks[-20:],
        "same_node_running_tasks": same_node_tasks[-30:],
    }


def _remember_running_resource_snapshots(state: dict, nodes: list):
    for task in state.get("tasks", []):
        if task.get("status") != "running":
            continue
        snap = _resource_snapshot_for_placement(
            state, nodes, task, task.get("node"), task.get("gpu_idx"),
            "running_probe",
        )
        task["last_resource_snapshot"] = snap


def _reconcile_aggregate_only_vram(state: dict, nodes: list) -> int:
    """Approximate task VRAM when aggregate GPU memory is known but per-PID VRAM is not."""
    changed = 0
    for node_state in nodes or []:
        if not node_state.get("alive"):
            continue
        node_name = node_state.get("name")
        for gpu in node_state.get("gpus") or []:
            gpu_idx = gpu.get("idx")
            try:
                aggregate_used = int(gpu.get("observed_used_mb", gpu.get("used_mb") or 0) or 0)
            except (TypeError, ValueError):
                continue
            tasks = [
                t for t in state.get("tasks", [])
                if t.get("status") == "running"
                and t.get("node") == node_name
                and t.get("gpu_idx") == gpu_idx
                and not _is_slurm_managed(t)
            ]
            if not tasks:
                continue
            baselines = []
            for t in tasks:
                snap = t.get("last_launch_pre_snapshot") or {}
                if snap.get("target_node") != node_name or snap.get("target_gpu_idx") != gpu_idx:
                    continue
                try:
                    baselines.append(int(((snap.get("gpu") or {}).get("used_mb")) or 0))
                except (TypeError, ValueError):
                    pass
            baseline = min(baselines) if baselines else 0
            known = 0
            unknown = []
            for t in tasks:
                cur = int(t.get("current_vram_mb") or 0)
                if cur > 0 and t.get("vram_estimation_source") != "aggregate_residual":
                    known += cur
                else:
                    unknown.append(t)
            if not unknown:
                continue
            residual = max(0, aggregate_used - baseline - known)
            if residual < 100:
                for t in unknown:
                    if t.get("vram_estimation_source") in ("aggregate_residual", "aggregate_observed_zero", None):
                        t["current_vram_mb"] = 0
                        t["peak_vram_mb"] = 0
                        t["vram_estimation_source"] = "aggregate_observed_zero"
                        t["vram_estimation_note"] = (
                            f"aggregate GPU memory on {node_name}:GPU{gpu_idx} has no residual for this task"
                        )
                        changed += 1
                continue
            weights = [max(1, int(t.get("est_vram_mb") or t.get("peak_vram_mb") or 1)) for t in unknown]
            total_weight = sum(weights) or len(unknown)
            remaining = residual
            for i, (t, weight) in enumerate(zip(unknown, weights)):
                if i == len(unknown) - 1:
                    share = remaining
                else:
                    share = int(round(residual * weight / total_weight))
                    remaining -= share
                if share < 100:
                    continue
                t["current_vram_mb"] = share
                t["peak_vram_mb"] = max(int(t.get("peak_vram_mb") or 0), share)
                t["vram_estimation_source"] = "aggregate_residual"
                t["vram_estimation_note"] = (
                    f"aggregate GPU memory attribution on {node_name}:GPU{gpu_idx}; "
                    "per-PID VRAM unavailable"
                )
                changed += 1
    return changed


def _last_progress_line(text: str) -> str:
    if not text:
        return ""
    progress_re = re.compile(
        r"(\d+\s*/\s*\d+\s*\[|\[\s*\d+\s*/\s*\d+\s*\]|"
        r"(?:^|[^\w])(?:Iter|Iteration|Epoch|Step|step)\s+\d+|"
        r"Starting training|JAX devices|Training complete|DONE)"
    )
    last = ""
    for line in text.splitlines():
        clean = line.replace("\r", "\n").splitlines()[-1] if "\r" in line else line
        if progress_re.search(clean):
            last = clean.strip()
    return last[-500:]


def _log_progress_forensics(task: dict, tail_text: str, elapsed_s: Optional[float] = None) -> dict:
    elapsed = float(elapsed_s if elapsed_s is not None else _effective_elapsed_s(task))
    out = {
        "last_progress_line": _last_progress_line(tail_text),
        "training_started": any(m in tail_text for m in ("Starting training", "Iter ", "Epoch ", "Step ")),
        "jax_device_seen": "JAX devices:" in tail_text,
    }
    et = _load_eta_tracker_module()
    if et:
        try:
            progress = et.parse_progress(tail_text, cmd=task.get("cmd"))
        except Exception:
            progress = None
        if progress:
            cur, total = progress
            out.update({
                "progress_current": int(cur),
                "progress_total": int(total),
                "progress_ratio": float(cur) / float(total) if total else None,
            })
        try:
            projection = et.runtime_projection(tail_text, elapsed_s=elapsed, cmd=task.get("cmd"))
        except Exception:
            projection = None
        if projection:
            out.update({
                "eta_source": projection.get("source"),
                "eta_seconds": projection.get("eta_s"),
                "runtime_total_s_est": projection.get("total_s"),
                "runtime_unit_s_est": projection.get("unit_s"),
            })
    if ("Failed to create stream executor" in tail_text
            or "Unable to initialize backend 'cuda'" in tail_text
            or "no supported devices found for platform CUDA" in tail_text):
        out["failure_stage"] = "cuda_init"
    elif out["training_started"] and not re.search(r"(?:^|\s)Iter\s+\d+", tail_text):
        out["failure_stage"] = "training_start_before_first_iter"
    elif out.get("progress_current") is not None or re.search(r"(?:^|\s)Iter\s+\d+", tail_text):
        out["failure_stage"] = "mid_training_or_after_progress"
    else:
        out["failure_stage"] = "pre_training_or_unknown"
    return out


def _build_crash_forensics_payload(task: dict, state: dict, nodes: Optional[list] = None) -> dict:
    diag = task.get("_diagnosis") or {}
    tail = diag.get("tail") or ""
    elapsed = int(diag.get("lifetime_s") or max(0, (task.get("finished_at") or time.time()) - (task.get("started_at") or time.time())))
    live_snapshot = _resource_snapshot_for_placement(
        state, nodes or [], task, task.get("node"), task.get("gpu_idx"),
        "crash_probe",
    ) if nodes else None
    last_snapshot = task.get("last_resource_snapshot") or task.get("last_launch_post_snapshot") or task.get("last_launch_pre_snapshot")
    progress = _log_progress_forensics(task, tail, elapsed_s=elapsed)
    payload = {
        "id": task.get("id"),
        "project": task.get("project"),
        "signature": task.get("signature"),
        "description": task.get("description", "")[:160],
        "node": task.get("node"),
        "gpu_idx": task.get("gpu_idx"),
        "status": task.get("status"),
        "lifetime_s": elapsed,
        "reason": diag.get("reason"),
        "log_path": diag.get("log_path") or task.get("log_path"),
        "log_size": diag.get("log_size"),
        "progress": progress,
        "per_task_vram_known": bool(int(task.get("peak_vram_mb") or 0) > 0),
        "peak_vram_mb": int(task.get("peak_vram_mb") or 0),
        "current_vram_mb": int(task.get("current_vram_mb") or 0),
        "last_resource_snapshot": last_snapshot,
        "crash_probe_snapshot": live_snapshot,
    }
    gpu_snap = None
    if isinstance(last_snapshot, dict):
        gpu_snap = last_snapshot.get("gpu")
    if not gpu_snap and isinstance(live_snapshot, dict):
        gpu_snap = live_snapshot.get("gpu")
    if gpu_snap:
        payload["gpu_over_one_third"] = bool(gpu_snap.get("over_one_third"))
        payload["gpu_over_full"] = bool(gpu_snap.get("over_full"))
        payload["gpu_used_mb"] = gpu_snap.get("used_mb")
        payload["gpu_total_mb"] = gpu_snap.get("total_mb")
        payload["gpu_one_third_mb"] = gpu_snap.get("one_third_mb")
    same_gpu = []
    if isinstance(last_snapshot, dict):
        same_gpu = last_snapshot.get("same_gpu_tasks") or []
    if not same_gpu and isinstance(live_snapshot, dict):
        same_gpu = live_snapshot.get("same_gpu_tasks") or []
    payload["same_gpu_tasks"] = same_gpu
    return payload


def _is_oom_like_forensics(payload: dict) -> bool:
    text = " ".join(str(payload.get(k) or "") for k in ("reason", "description"))
    progress = payload.get("progress") or {}
    text += " " + str(progress.get("failure_stage") or "")
    return any(s in text.lower() for s in ("oom", "out of memory", "cuda_error_out_of_memory", "resource_exhausted"))


def _build_oom_forensics_payload(crash_payload: dict, state: dict) -> dict:
    task_id = crash_payload.get("id")
    same_gpu = crash_payload.get("same_gpu_tasks") or []
    trigger = None
    victim_started = 0.0
    for t in same_gpu:
        if t.get("id") == task_id:
            victim_started = float(t.get("started_at") or 0)
            break
    candidates = [t for t in same_gpu if t.get("id") != task_id]
    if candidates:
        # Prefer a task that arrived around/after the victim; otherwise use the latest colocated task.
        recent = [t for t in candidates if float(t.get("started_at") or 0) >= victim_started - 60]
        trigger = max(recent or candidates, key=lambda x: float(x.get("started_at") or 0))
    by_id = {t.get("id"): t for t in state.get("tasks", [])}
    trigger_status = None
    if trigger:
        cur = by_id.get(trigger.get("id"))
        trigger_status = cur.get("status") if cur else trigger.get("status")
    return {
        "id": task_id,
        "node": crash_payload.get("node"),
        "gpu_idx": crash_payload.get("gpu_idx"),
        "lifetime_s": crash_payload.get("lifetime_s"),
        "failure_stage": (crash_payload.get("progress") or {}).get("failure_stage"),
        "over_one_third": crash_payload.get("gpu_over_one_third"),
        "over_full": crash_payload.get("gpu_over_full"),
        "exact_per_task_vram_known": crash_payload.get("per_task_vram_known"),
        "gpu_used_mb": crash_payload.get("gpu_used_mb"),
        "gpu_total_mb": crash_payload.get("gpu_total_mb"),
        "gpu_one_third_mb": crash_payload.get("gpu_one_third_mb"),
        "suspected_trigger_task": trigger,
        "trigger_task_status_after_oom": trigger_status,
        "oom_victim_status": by_id.get(task_id, {}).get("status"),
        "reason": crash_payload.get("reason"),
        "log_path": crash_payload.get("log_path"),
    }


def _task_is_gpu_capacity_task(task: dict) -> bool:
    try:
        need_vram = int(task.get("est_vram_mb", DEFAULT_VRAM_MB) or 0)
    except Exception:
        need_vram = int(DEFAULT_VRAM_MB or 0)
    return need_vram > 0 and not bool(task.get("cpu_fallback_selected"))


def _flag_enabled(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "no", "off", "none")
    return bool(value)


def _ignore_cpu_for_server_gpu_task(task: dict, node_state: Optional[dict] = None,
                                    node_info: Optional[dict] = None,
                                    node_name: Optional[str] = None,
                                    gpu_idx: Optional[int] = None) -> bool:
    """Remote Linux GPU jobs are GPU/RAM placed; CPU pressure only gates local and CPU-only work."""
    if not _task_is_gpu_capacity_task(task):
        return False
    node_name = node_name or (node_state or {}).get("name") or task.get("node")
    if not node_name or node_name == "local" or _node_is_windows(node_name):
        return False
    info = node_info if node_info is not None else NODES.get(node_name, {})
    if not (info or {}).get("host"):
        return False
    if (info or {}).get("max_vram_per_task") == 0:
        return False
    if node_state is not None:
        if not node_state.get("gpus"):
            return False
    elif gpu_idx is None and task.get("gpu_idx") is None:
        return False
    return _flag_enabled((info or {}).get("ignore_cpu_for_gpu_tasks"), True)


def _task_ignores_one_third_pack_rule(task: dict, node_info: Optional[dict] = None) -> bool:
    if _flag_enabled((task or {}).get("allow_gpu_over_one_third"), False):
        return True
    if str((task or {}).get("signature") or "").startswith("BAPR/bus_v2/"):
        return True
    return _flag_enabled((node_info or {}).get("allow_gpu_over_one_third"), False)


def _task_required_gpu_idx(task: dict) -> Optional[int]:
    raw = task.get("require_gpu_idx")
    if raw is None or raw == "":
        return None
    try:
        idx = int(raw)
    except Exception:
        return None
    return idx if idx >= 0 else None


def _node_numeric(node_state: Optional[dict], *keys, default=None):
    for key in keys:
        if not node_state or key not in node_state:
            continue
        val = node_state.get(key)
        if val is None or val == "":
            continue
        try:
            return float(val)
        except Exception:
            continue
    return default


def _local_cpu_pressure_high(node_state: Optional[dict], threshold_pct: int) -> tuple[bool, str]:
    """True when local is actually CPU pressured, not merely over-reserved.

    For WSL local, loadavg can look full while Windows host CPU still has real
    headroom. Prefer the Windows host CPU percentage when present; fall back to
    WSL/loadavg only if the host probe is unavailable.
    """
    host_cpu = _node_numeric(node_state, "host_cpu_load_pct")
    if host_cpu is not None:
        high = host_cpu >= threshold_pct
        return high, f"host_cpu={int(host_cpu)}% threshold={threshold_pct}%"
    total = int(_node_numeric(node_state, "total_cpu", default=0) or 0)
    free = _node_numeric(node_state, "observed_free_cpu", "wsl_free_cpu", "free_cpu")
    load = _node_numeric(node_state, "observed_loadavg", "wsl_loadavg", "loadavg")
    if total <= 0:
        return False, "cpu pressure unknown"
    if free is not None:
        high = int(free) <= 0
        return high, f"free_cpu={int(free)}/{total}"
    if load is not None:
        high = float(load) >= max(1, total)
        return high, f"load={float(load):.1f}/{total}"
    return False, "cpu pressure unknown"


def _local_gpu_cpu_block_reason(task: dict, node_state: Optional[dict]) -> str:
    if (node_state or {}).get("name") != "local":
        return ""
    if not _task_is_gpu_capacity_task(task):
        return ""
    high, why = _local_cpu_pressure_high(node_state, LOCAL_GPU_HOST_CPU_BLOCK_PCT)
    if high:
        return f"host cpu pressure: {why}"
    return ""


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
    if not _ignore_cpu_for_server_gpu_task(task, node_state, node_info):
        local_gpu_block = _local_gpu_cpu_block_reason(task, node_state)
        if local_gpu_block:
            return False, local_gpu_block
        if not ((node_state or {}).get("name") == "local" and _task_is_gpu_capacity_task(task)):
            needed_cpu = task.get("cpu_cores", DEFAULT_CPU_CORES)
            raw_free_cpu = int(node_state.get("free_cpu", 0) or 0)
            reserved_cpu = max(0, int((node_info or {}).get("reserved_cpu_cores") or 0))
            sched_free_cpu = max(0, raw_free_cpu - reserved_cpu)
            if sched_free_cpu < needed_cpu:
                reserve_note = f" (reserve {reserved_cpu})" if reserved_cpu else ""
                return False, f"cpu: need {needed_cpu}, free {sched_free_cpu}/{node_state.get('total_cpu', '?')}{reserve_note}"
    needed_ram = task.get("ram_mb", DEFAULT_RAM_MB)
    headroom = _node_ram_headroom_mb(node_state, node_info)
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


def _task_capability_text(task: dict) -> str:
    return "\n".join(str(task.get(k) or "") for k in (
        "cmd", "cwd", "description", "signature", "project"
    )).lower()


def _task_gpu_capability(task: dict) -> str:
    """Classify the GPU runtime a task needs for node compatibility checks."""
    text = _task_capability_text(task)
    if any(p in text for p in ("resac-jax", "jax_experiments", "xla_python_client", "xla_flags")):
        return "jax_cuda"
    if any(p in text for p in ("torch", "pytorch", "sac_ensemble_original_logging.py")):
        return "torch_cuda"
    return "cuda"


def _task_cpu_fallback_capability(task: dict) -> str:
    gpu_cap = _task_gpu_capability(task)
    if gpu_cap == "jax_cuda":
        return "jax_cpu"
    if gpu_cap == "torch_cuda":
        return "torch_cpu"
    return "cpu"


def _node_cpu_fallback_block_reason(task, node_name, node_info):
    """Return why a GPU task cannot opportunistically run CPU-only on this node."""
    if (task.get("est_vram_mb") or 0) <= 0:
        return None
    if not bool(task.get("allow_cpu_training", False)):
        return (
            "cpu-fallback disabled: task has vram>0 and is treated as GPU-required; "
            "resubmit with --allow-cpu-training only if CPU training is intentional"
        )
    caps = set(str(c).lower() for c in ((node_info or {}).get("capabilities") or []))
    need = _task_cpu_fallback_capability(task)
    if caps and need not in caps:
        return f"cpu-fallback capability block: need {need}"
    return None


def _clear_disallowed_cpu_fallback_selection(task: dict) -> bool:
    """Drop stale CPU fallback placement left by older watcher versions."""
    if int(task.get("est_vram_mb") or 0) <= 0:
        return False
    if not task.get("cpu_fallback_selected"):
        return False
    if bool(task.get("allow_cpu_training", False)):
        return False
    task.pop("cpu_fallback_selected", None)
    task.pop("cpu_fallback_original_vram_mb", None)
    task.pop("cpu_fallback_capability", None)
    return True


def _task_launch_cpu_mode(task: dict) -> bool:
    if int(task.get("est_vram_mb") or 0) <= 0:
        return True
    return bool(task.get("cpu_fallback_selected")) and bool(task.get("allow_cpu_training", False))


def _node_gpu_task_block_reason(task, node_name, node_info):
    """Return a node policy reason if this GPU task should not run on node_name."""
    if (task.get("est_vram_mb") or 0) <= 0:
        return None
    caps = set(str(c).lower() for c in ((node_info or {}).get("capabilities") or []))
    need = _task_gpu_capability(task)
    if caps and need not in caps and "cuda" not in caps:
        return f"gpu capability block: need {need}"
    if caps and need not in caps and need != "cuda":
        return f"gpu capability block: need {need}"
    patterns = (node_info or {}).get("blocked_gpu_cmd_patterns") or []
    if not patterns:
        return None
    text = _task_capability_text(task)
    for pattern in patterns:
        pat = str(pattern or "").strip()
        if pat and pat.lower() in text:
            return f"gpu-policy block: matches {pat}"
    return None


def _gpu_fits(task, gpu, node_info):
    """VRAM + compute-saturation check on a specific GPU.

    Policy: on an empty/noise-only GPU, one large task may exceed the freeze line;
    on an occupied GPU, do not place a task if the GPU is already at the 1/3
    memory line plus a small grace window, or this placement would cross it.
    No small-task exemption. This freezes warm cards until old work completes,
    because dispatching into a
    partially occupied JAX/PyTorch card is exactly the scenario that makes OOM
    forensics ambiguous and can kill useful in-flight progress.
    """
    cap = node_info.get("max_vram_per_task")
    if cap is not None and task["est_vram_mb"] > cap:
        return False
    if ONE_THIRD_PACK_RULE and not _task_ignores_one_third_pack_rule(task, node_info):
        freeze = _gpu_freeze_line_mb(int(gpu.get("total_mb") or 0))
        used = int(gpu.get("used_mb") or 0)
        need = int(task.get("est_vram_mb") or 0)
        if freeze > 0 and used > 100 and (used >= freeze or used + need >= freeze):
            return False
    # Compute saturation: if there's already a task on this GPU and it's pinning the chip,
    # don't pack more — the new task would just steal cycles and slow everyone down.
    # The "occupied" guard (>100MB) avoids blocking on a transient util spike on an empty GPU.
    util_limit = _node_gpu_util_limit(node_info)
    if util_limit is not None and gpu["used_mb"] > 100 and gpu.get("util_pct", 0) >= util_limit:
        return False
    if gpu["free_mb"] < task["est_vram_mb"] + VRAM_MARGIN_MB:
        return False
    if _algorithm_gpu_fit_block_reason(task, gpu, node_info):
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
    allowed_nodes = task.get("allowed_nodes") or []
    if allowed_nodes:
        allowed = {str(n) for n in allowed_nodes}
        nodes = [n for n in nodes if n.get("name") in allowed]
    preferred = task.get("preferred_node")
    require = task.get("require_node")  # HARD pin — never falls back
    require_gpu_idx = _task_required_gpu_idx(task)
    resume_preferred = []
    for n in (task.get("resume_preferred_nodes") or []):
        if n and n not in resume_preferred:
            resume_preferred.append(n)
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
        if _evict_node_cooldown_block_reason(task, n["name"], node_state=n):
            return []
        node_info = NODES[n["name"]]
        if node_info.get("only_when_targeted"):
            targeted = (
                require == n["name"]
                or preferred == n["name"]
                or n["name"] in (task.get("allowed_nodes") or [])
                or _task_requests_slurm(task)
            )
            if not targeted:
                return []
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
        if not _requires_local_capacity_check(n["name"], task, n):
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
        cpu_fallback = False
        if not cpu_only:
            cpu_fallback = (
                _node_is_windows(n["name"])
                or not n.get("gpus")
                or node_info.get("max_vram_per_task") == 0
            )
            if cpu_fallback and _node_cpu_fallback_block_reason(task, n["name"], node_info):
                return []
        if _node_is_windows(n["name"]) and not (cpu_only or cpu_fallback):
            return []
        if _task_requests_slurm(task):
            # The user supplied Slurm-only fields, so do not silently discard
            # them by launching through LocalBackend on a non-Slurm/default-local
            # node. If no Slurm-capable/Slurm-routed node exists, the task stays queued
            # with an explicit reason instead of running with the wrong policy.
            return []
        if not cpu_only and _node_gpu_task_block_reason(task, n["name"], node_info):
            if not cpu_fallback:
                return []
        ok, _why = _node_resources_ok(task, n, node_info)
        if not ok: return []
        out = []
        if cpu_only or cpu_fallback:
            node_info = NODES.get(n["name"], {})
            cpu_labor_bonus = -1 if node_info.get("cpu_labor_node") else 0
            fallback_penalty = 2 if cpu_fallback and not cpu_only else 0
            rt = _candidate_runtime_seconds(task, n["name"], None)
            rt_unknown = 1 if rt <= 0 else 0
            score = (fallback_penalty, rt_unknown, rt, cpu_labor_bonus, -n["free_cpu"], -n["free_ram_mb"])
            out.append((score, n["name"], None))
        else:
            for g in n["gpus"]:
                try:
                    g_idx = int(g.get("idx"))
                except Exception:
                    continue
                if require_gpu_idx is not None and g_idx != require_gpu_idx:
                    continue
                max_tasks_per_gpu = node_info.get("max_tasks_per_gpu")
                if max_tasks_per_gpu is not None:
                    try:
                        gpu_task_count = int(g.get("running_task_count") or 0)
                        gpu_task_cap = int(max_tasks_per_gpu)
                    except Exception:
                        gpu_task_count = 0
                        gpu_task_cap = 0
                    if gpu_task_cap > 0 and gpu_task_count >= gpu_task_cap:
                        continue
                if not _gpu_fits(task, g, node_info): continue
                # Empty-first placement: if a node has a genuinely idle GPU, use it before
                # stacking onto a warm card. Among warm fitting cards, prefer the lowest
                # post-placement memory pressure. With the 1/3+grace freeze rule, packing
                # the fullest warm card first just pushes it to the freeze line while a
                # cooler sibling stays underused.
                fits_remaining = g["free_mb"] - (task.get("est_vram_mb") or 0)
                used = int(g.get("used_mb") or 0)
                total = max(1, int(g.get("total_mb") or 1))
                need = int(task.get("est_vram_mb") or 0)
                occupied = 1 if used >= GPU_EMPTY_USED_MB else 0
                rt = _candidate_runtime_seconds(task, n["name"], g["idx"])
                rt_unknown = 1 if rt <= 0 else 0
                if occupied:
                    score = (occupied, rt_unknown, rt, float(used + need) / float(total), used + need)
                else:
                    score = (occupied, rt_unknown, rt, fits_remaining)
                score = _algorithm_gpu_score(task, n, g, score)
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

        # 1) Resume locality wins over ordinary soft preference. If a
        # checkpoint already exists on a server, launching there avoids
        # silent step-0 restarts and large checkpoint rsyncs.
        tried = set()
        for resume_node in resume_preferred:
            for n in search_nodes:
                if n["name"] != resume_node:
                    continue
                tried.add(n["name"])
                cands = _candidates_for_node(n)
                if cands:
                    cands.sort()
                    return cands[0][1], cands[0][2]

        # 2) Try preferred node next (if specified and alive).
        if preferred:
            for n in search_nodes:
                if n["name"] == preferred:
                    tried.add(n["name"])
                    cands = _candidates_for_node(n)
                    if cands:
                        cands.sort()
                        return cands[0][1], cands[0][2]
                    # preferred is alive but full — fall through to fallback search

        # 3) Fallback: scan all nodes (excluding nodes already tried above).
        cands = []
        for n in search_nodes:
            if n["name"] in tried:
                continue
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
    if _node_is_windows(task["node"]):
        warnings.append(
            f"warn: git precheck skipped on Windows node {task['node']} "
            "(path mapping / git availability is project-specific)")
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
RESUME_SCAN_TTL_S = int(os.environ.get("SCHEDULEURM_RESUME_SCAN_TTL_S", "120"))
RESUME_SCAN_WORKERS = max(1, int(os.environ.get("SCHEDULEURM_RESUME_SCAN_WORKERS", "4")))

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

def _resume_node_names(task: Optional[dict] = None, nodes: Optional[list] = None) -> list:
    """Configured nodes whose filesystem can be checked for this task's checkpoints.

    Keep resume scans scoped to nodes the task could actually use. Bulk RE-SAC
    queues can contain hundreds of resume-capable tasks; probing every node for
    every task serially starves dispatch even when only a few GPU slots are open.
    """
    names = []
    live_names = None
    if nodes is not None:
        live_names = {n.get("name") for n in nodes if n.get("name") and n.get("alive")}

    def add(name, *, force: bool = False):
        if not name or name not in NODES or name in names:
            return
        if live_names is not None and name not in live_names and not force:
            return
        names.append(name)

    known_resume_nodes = []
    if task:
        for loc in (task.get("resume_locations") or []):
            node = loc.get("node") if isinstance(loc, dict) else None
            if node and node not in known_resume_nodes:
                known_resume_nodes.append(node)
        for key in ("resume_checkpoint_node", "staged_node", "node", "assigned_node"):
            node = task.get(key)
            if node and node not in known_resume_nodes:
                known_resume_nodes.append(node)

        explicit = []
        require_node = task.get("require_node")
        if require_node:
            explicit.append(require_node)
        for node in (task.get("allowed_nodes") or []):
            if node and node not in explicit:
                explicit.append(node)

        for node in explicit:
            add(str(node))
        # A known checkpoint location is safety-critical. Keep it in the scan
        # set even if the current probe marks the node down, so dispatch does
        # not silently restart elsewhere while the old checkpoint is unavailable.
        for node in known_resume_nodes:
            add(str(node), force=True)
        if explicit and names:
            return names
        # preferred_node and resume_preferred_nodes are soft placement hints.
        # Scan them first, but continue to fallback nodes because pick_placement
        # may legally place elsewhere if the preferred node is full.
        preferred = []
        for node in [task.get("preferred_node")] + list(task.get("resume_preferred_nodes") or []):
            if node and node not in preferred:
                preferred.append(node)
        for node in preferred:
            add(str(node))

    fallback = [n.get("name") for n in (nodes or []) if n.get("name")] or list(NODES.keys())
    gpu_task = bool(task and int(task.get("est_vram_mb") or 0) > 0)
    need_cap = _task_gpu_capability(task or {}) if gpu_task else None
    for name in fallback:
        info = NODES.get(name, {})
        if gpu_task:
            if _node_is_windows(name) or info.get("max_vram_per_task") == 0:
                continue
            caps = set(str(c).lower() for c in (info.get("capabilities") or []))
            if caps and need_cap not in caps and "cuda" not in caps:
                continue
        add(name)
    return names


def _resume_scan_key(task: dict, nodes: Optional[list] = None) -> list:
    return [
        task.get("ckpt_dir"),
        task.get("ckpt_glob", "*") or "*",
        sorted(_resume_node_names(task, nodes)),
    ]


def _task_requires_resume_scan(task: dict) -> bool:
    """A resume-capable task must have checkpoint existence checked before launch."""
    if task.get("skip_resume_scan"):
        return False
    if not task.get("ckpt_dir"):
        return False
    return bool(
        task.get("resume_flag")
        or task.get("resume_managed_by_cmd")
        or _cmd_has_resume_flag(task.get("cmd") or "")
    )


def _find_resume_on_node(task: dict, node: str) -> Optional[dict]:
    """Latest checkpoint in task['ckpt_dir'] on one node, with metadata."""
    ckpt_dir = task.get("ckpt_dir")
    if not ckpt_dir:
        return None
    if _node_is_windows(node):
        win_dir = _windows_path_for_node(node, ckpt_dir)
        pattern = task.get("ckpt_glob", "*") or "*"
        ext_list = ",".join(CKPT_EXTS)
        safe_pat = RESUME_SAFE_NAME_RE.pattern
        unsafe_pat = RESUME_UNSAFE_NAME_RE.pattern
        explicit = "1" if pattern != "*" else "0"
        ps = rf'''
$dir = {_ps_quote(win_dir)}
$pattern = {_ps_quote(pattern)}
$explicit = ({_ps_quote(explicit)} -eq '1')
$exts = ({_ps_quote(ext_list)}).Split(',')
$safe = {_ps_quote(safe_pat)}
$unsafe = {_ps_quote(unsafe_pat)}
if (-not (Test-Path -LiteralPath $dir)) {{ exit 0 }}
$files = Get-ChildItem -LiteralPath $dir -File -Filter $pattern -ErrorAction SilentlyContinue | Where-Object {{
  $ext = $_.Extension.TrimStart('.').ToLowerInvariant()
  if ($exts -notcontains $ext) {{ return $false }}
  if ($explicit) {{ return $true }}
  if ($_.Name -match $unsafe) {{ return $false }}
  return ($_.Name -match $safe)
}} | Sort-Object LastWriteTimeUtc -Descending
if ($files -and $files.Count -gt 0) {{
  $f = $files[0]
  [pscustomobject]@{{ path=$f.FullName; mtime=([DateTimeOffset]$f.LastWriteTimeUtc).ToUnixTimeSeconds(); size=$f.Length }} | ConvertTo-Json -Compress
}}
'''
        try:
            rc, out, err = _run_windows_ps(node, ps, timeout=10, check=False)
        except Exception as e:
            raise RuntimeError(str(e))
        if rc != 0:
            raise RuntimeError((err or out or f"rc={rc}").strip()[:300])
        out = out.strip()
        if not out:
            return None
        try:
            data = json.loads(out.splitlines()[-1])
        except Exception as e:
            raise RuntimeError(f"bad Windows checkpoint scan output: {e}")
        data["node"] = node
        return data
    scan_dir = _remote_path_for_node(node, ckpt_dir)
    pattern = task.get("ckpt_glob", "*") or "*"
    explicit_glob = pattern != "*"
    script = r'''
import glob, json, os, re, sys
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
    p = paths[0]
    print(json.dumps({"path": p, "mtime": os.path.getmtime(p), "size": os.path.getsize(p)}))
'''
    marker_begin = "__SCHEDULEURM_RESUME_SCAN_JSON_BEGIN__"
    marker_end = "__SCHEDULEURM_RESUME_SCAN_JSON_END__"
    py = str((NODES.get(node, {}) or {}).get("resume_scan_python") or "python3")
    cmd = (
        f"printf '%s\\n' {shlex.quote(marker_begin)}; "
        f"{shlex.quote(py)} - {shlex.quote(scan_dir)} {shlex.quote(pattern)} "
        f"{'1' if explicit_glob else '0'} {shlex.quote(','.join(CKPT_EXTS))} "
        f"{shlex.quote(RESUME_SAFE_NAME_RE.pattern)} {shlex.quote(RESUME_UNSAFE_NAME_RE.pattern)}"
        f" <<'PY'\n{script}\nPY\n"
        f"rc=$?; printf '%s\\n' {shlex.quote(marker_end)}; exit $rc"
    )
    try:
        rc, out, err = run_on(node, cmd, timeout=10, check=False)
    except Exception as e:
        raise RuntimeError(str(e))
    if rc != 0:
        raise RuntimeError((err or out or f"rc={rc}").strip()[:300])
    text = out or ""
    if marker_begin in text and marker_end in text:
        text = text.split(marker_begin, 1)[1].split(marker_end, 1)[0]
    out = text.strip()
    if not out:
        return None
    try:
        data = json.loads(out.splitlines()[-1])
    except Exception as e:
        raise RuntimeError(f"bad checkpoint scan output: {e}")
    data["node"] = node
    return data


def scan_resume_locations(task: dict, nodes: Optional[list] = None, cache: Optional[dict] = None) -> tuple:
    """Check all alive scheduler nodes for an existing resume checkpoint.

    Returns (locations, errors). Locations are sorted newest-first and contain
    {node, path, mtime, size}. The scan is keyed by ckpt_dir/glob so a dispatch
    cycle checks shared paths once even if multiple queued records reference it.
    """
    if not _task_requires_resume_scan(task):
        return [], {}
    node_names = _resume_node_names(task, nodes)
    key = (task.get("ckpt_dir"), task.get("ckpt_glob", "*") or "*", tuple(node_names))
    if cache is not None and key in cache:
        locs, errs = cache[key]
        return list(locs), dict(errs)
    locations = []
    errors = {}
    def _scan_one(node):
        try:
            return node, _find_resume_on_node(task, node), None
        except Exception as e:
            return node, None, str(e)[:300]

    workers = min(RESUME_SCAN_WORKERS, len(node_names))
    if workers <= 1 or len(node_names) <= 1:
        results = [_scan_one(node) for node in node_names]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_scan_one, node): node for node in node_names}
            for fut in as_completed(futures):
                results.append(fut.result())
    for node, found, err in results:
        if err:
            errors[node] = err
        elif found:
            locations.append(found)
    locations.sort(key=lambda x: (float(x.get("mtime") or 0), str(x.get("node") or "")), reverse=True)
    if cache is not None:
        cache[key] = (list(locations), dict(errors))
    return locations, errors


def _refresh_resume_locations_for_task(task: dict, nodes: list, cache: dict) -> tuple:
    """Persist checkpoint scan results on the task and expose node preference."""
    if not _task_requires_resume_scan(task):
        for k in ("resume_locations", "resume_scan_errors", "resume_preferred_nodes",
                  "resume_checkpoint_node", "resume_scan_at", "resume_scan_key"):
            task.pop(k, None)
        return [], {}
    key = _resume_scan_key(task, nodes)
    now = time.time()
    last_scan = float(task.get("resume_scan_at") or 0)
    if (
        "resume_locations" in task
        and task.get("resume_scan_key") == key
        and last_scan > 0
        and now - last_scan <= RESUME_SCAN_TTL_S
    ):
        return list(task.get("resume_locations") or []), dict(task.get("resume_scan_errors") or {})
    locations, errors = scan_resume_locations(task, nodes=nodes, cache=cache)
    task["resume_scan_at"] = now
    task["resume_scan_key"] = key
    task["resume_locations"] = locations
    if errors:
        task["resume_scan_errors"] = errors
    else:
        task.pop("resume_scan_errors", None)
    if locations:
        ordered_nodes = []
        for loc in locations:
            node = loc.get("node")
            if node and node not in ordered_nodes:
                ordered_nodes.append(node)
        task["resume_preferred_nodes"] = ordered_nodes
        best = locations[0]
        task["resume_checkpoint_node"] = best.get("node")
        task["resume_from"] = best.get("path")
    else:
        task.pop("resume_preferred_nodes", None)
        task.pop("resume_checkpoint_node", None)
        task.pop("resume_from", None)
    return locations, errors


def _resume_location_for_node(task: dict, node: str) -> Optional[dict]:
    for loc in task.get("resume_locations") or []:
        if loc.get("node") == node:
            return loc
    return None


def find_resume(task):
    """Latest checkpoint in task['ckpt_dir'] on target node, by mtime. None if no dir / no files.

    Filters results by extension whitelist of known torch/tf/jax/numpy ckpt formats. Without
    this, default glob `*` matched ANY file (e.g. train_log.csv) which then got passed as
    --resume_from <path> → torch.load() blew up with EOFError.

    With the default glob, also require checkpoint-looking names and skip common output artifacts
    (`model_final.pt`, `buffer*.pkl`, metrics/results/eval dumps). Those files can have checkpoint
    extensions but often lack optimizer/RNG/replay state and are not safe training resume targets.
    A non-default --ckpt-glob is treated as explicit user intent and only uses the extension filter."""
    node = task.get("node")
    if not node:
        return None
    try:
        found = _find_resume_on_node(task, node)
    except Exception:
        return None
    return (found or {}).get("path") or None

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

    def requires_local_capacity_check(self, node: str, task: Optional[dict] = None,
                                      node_state: Optional[dict] = None) -> bool:
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


def _ps_quote(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def _windows_path_for_node(node: str, path: str) -> str:
    r"""Map common local Linux workspace paths to jtl110cpu's Windows layout.

    The manual jtl110cpu workflow used F:\<project>. Keep this mapping
    conservative and transparent; callers can always submit an explicit
    Windows path (`F:\foo`) and it will be left untouched.
    """
    if not _node_is_windows(node) or not path:
        return path
    text = str(path)
    if re.match(r"^[A-Za-z]:[\\/]", text) or text.startswith("\\\\"):
        return text.replace("/", "\\")
    norm = os.path.normpath(text)
    root = (NODES.get(node, {}) or {}).get("windows_workspace_root") or r"F:\\"
    root = root.rstrip("\\/") + "\\"
    prefixes = [
        str(Path.home() / "mine_code"),
        str(Path.home()),
        "/home/erzhu419/mine_code",
        "/home/erzhu419",
    ]
    for pref in prefixes:
        pref = os.path.normpath(pref)
        try:
            if os.path.commonpath([norm, pref]) != pref:
                continue
        except Exception:
            continue
        rel = os.path.relpath(norm, pref)
        parts = rel.split(os.sep)
        if not parts or parts[0] in (".", ".."):
            continue
        # /home/u/mine_code/project/subdir -> F:\project\subdir
        return root + "\\".join(parts)
    return text.replace("/", "\\")


def _remote_path_for_node(node: str, path: str) -> str:
    """Map a local POSIX workspace path to a node-specific remote path.

    Linux GPU nodes share the same user/path convention as local, so they leave
    paths unchanged. Campus/HPC accounts often have a different home directory;
    those nodes can declare remote_workspace_root + remote_path_prefixes.
    """
    if not path:
        return path
    if _node_is_windows(node):
        return _windows_path_for_node(node, path)
    info = NODES.get(node, {}) or {}
    root = str(info.get("remote_workspace_root") or "").strip()
    if not root:
        return path
    text = str(path)
    if not text.startswith("/"):
        return text
    norm = os.path.normpath(text)
    prefixes = list(info.get("remote_path_prefixes") or [])
    if not prefixes:
        prefixes = [
            str(Path.home() / "mine_code"),
            "/home/erzhu419/mine_code",
        ]
    root = root.rstrip("/")
    for pref in prefixes:
        pref = os.path.normpath(os.path.expanduser(str(pref)))
        try:
            if os.path.commonpath([norm, pref]) != pref:
                continue
        except Exception:
            continue
        rel = os.path.relpath(norm, pref)
        if not rel or rel == ".":
            return root
        parts = rel.split(os.sep)
        if parts[0] == "..":
            continue
        return root + "/" + "/".join(parts)
    return text


def _rewrite_command_paths_for_node(node: str, cmd: str) -> str:
    """Rewrite configured local workspace prefixes inside a shell command."""
    if not node or not cmd or _node_is_windows(node):
        return cmd
    info = NODES.get(node, {}) or {}
    if not info.get("remote_workspace_root"):
        return cmd
    out = str(cmd)
    for pref in (info.get("remote_path_prefixes") or []):
        pref = os.path.normpath(os.path.expanduser(str(pref)))
        mapped = _remote_path_for_node(node, pref)
        if mapped and mapped != pref:
            out = out.replace(pref, mapped)
    return out


def _windows_rewrite_token(node: str, token: str) -> str:
    if not token:
        return token
    # Python executable paths are almost never portable from Linux/WSL to the
    # Windows CPU node. Use the node's configured Python for command leaders.
    base = os.path.basename(token.replace("\\", "/")).lower()
    if base in ("python", "python.exe", "python3", "python3.exe") or base.startswith("python3."):
        return (NODES.get(node, {}) or {}).get("windows_python") or token
    if token.startswith("/home/") or token.startswith(str(Path.home())):
        return _windows_path_for_node(node, token)
    return token


def _windows_prepare_command(task: dict) -> dict:
    """Return a wrapper payload command for WindowsBackend.

    Simple commands are passed as argv so Windows quoting is handled by
    subprocess. Shell-shaped commands fall back to `cmd /c` after path
    translation; this keeps wrappers like `python eval.py ...` robust while
    still allowing advanced user commands when needed.
    """
    node = task.get("node")
    inner = _inject_python_u(task.get("cmd") or "")
    resume_path = task.get("resume_from")
    resume_flag = task.get("resume_flag") or ""
    if resume_path and resume_flag:
        inner = f"{inner} {resume_flag} {shlex.quote(resume_path)}"
    shell_meta = bool(re.search(r"[;&|<>]", inner))
    if not shell_meta:
        try:
            toks = shlex.split(inner)
        except Exception:
            toks = []
        if toks:
            env = {}
            while toks and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", toks[0]):
                k, v = toks.pop(0).split("=", 1)
                if k.upper() == "PYTHONPATH":
                    parts = [_windows_path_for_node(node, p) for p in v.split(":") if p]
                    v = ";".join(parts)
                elif v.startswith("/home/") or v.startswith(str(Path.home())):
                    v = _windows_path_for_node(node, v)
                env[k] = v
            if not toks:
                return {"argv": [], "env": env}
            toks[0] = _windows_rewrite_token(node, toks[0])
            toks = [_windows_rewrite_token(node, t) for t in toks]
            if _task_launch_cpu_mode(task):
                device_aware = "--device" in toks or any("jax_experiments" in tok for tok in toks)
                if "--device" in toks:
                    idx = toks.index("--device")
                    if idx + 1 < len(toks):
                        toks[idx + 1] = "cpu"
                elif device_aware:
                    toks.extend(["--device", "cpu"])
            return {"argv": toks, "env": env}
    mapped = inner
    for pref in (str(Path.home() / "mine_code"), str(Path.home()), "/home/erzhu419/mine_code", "/home/erzhu419"):
        if pref in mapped:
            mapped = mapped.replace(pref, _windows_path_for_node(node, pref))
    py = (NODES.get(node, {}) or {}).get("windows_python") or "python"
    mapped = re.sub(r"^(python(?:3(?:\.\d+)?)?(?:\.exe)?)(\s|$)",
                    lambda m: py + m.group(2),
                    mapped,
                    count=1,
                    flags=re.IGNORECASE)
    return {"cmdline": mapped}


def _windows_env_spec_error(task: dict) -> Optional[str]:
    spec = (task.get("env_spec") or "none").strip()
    low = spec.lower()
    if low in ("", "none", "auto"):
        return None
    if low.startswith("docker") or low.startswith("conda"):
        return (
            f"WindowsBackend does not support explicit env_spec={spec!r}; "
            "preinstall/use the node's configured windows_python, or run this "
            "task on a Linux node where docker/conda env sync is available."
        )
    return f"unsupported env_spec={spec!r} for WindowsBackend"


_WINDOWS_LAUNCHER = r'''
import base64, ctypes, json, os, subprocess, sys, time
from ctypes import wintypes

CREATE_NEW_PROCESS_GROUP = 0x00000200
TH32CS_SNAPPROCESS = 0x00000002
TH32CS_SNAPTHREAD = 0x00000004
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
THREAD_SET_INFORMATION = 0x0020

k32 = ctypes.windll.kernel32

class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.c_void_p),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", wintypes.WCHAR * 260),
    ]

class THREADENTRY32(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ThreadID", wintypes.DWORD),
        ("th32OwnerProcessID", wintypes.DWORD),
        ("tpBasePri", ctypes.c_long),
        ("tpDeltaPri", ctypes.c_long),
        ("dwFlags", wintypes.DWORD),
    ]

class GROUP_AFFINITY(ctypes.Structure):
    _fields_ = [
        ("Mask", ctypes.c_size_t),
        ("Group", wintypes.WORD),
        ("Reserved", wintypes.WORD * 3),
    ]

k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
k32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
k32.Process32FirstW.restype = wintypes.BOOL
k32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
k32.Process32NextW.restype = wintypes.BOOL
k32.Thread32First.argtypes = [wintypes.HANDLE, ctypes.POINTER(THREADENTRY32)]
k32.Thread32First.restype = wintypes.BOOL
k32.Thread32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(THREADENTRY32)]
k32.Thread32Next.restype = wintypes.BOOL
k32.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
k32.OpenThread.restype = wintypes.HANDLE
k32.SetThreadGroupAffinity.argtypes = [wintypes.HANDLE, ctypes.POINTER(GROUP_AFFINITY), ctypes.POINTER(GROUP_AFFINITY)]
k32.SetThreadGroupAffinity.restype = wintypes.BOOL
k32.CloseHandle.argtypes = [wintypes.HANDLE]
k32.CloseHandle.restype = wintypes.BOOL
k32.GetActiveProcessorGroupCount.argtypes = []
k32.GetActiveProcessorGroupCount.restype = wintypes.WORD
k32.GetActiveProcessorCount.argtypes = [wintypes.WORD]
k32.GetActiveProcessorCount.restype = wintypes.DWORD

def process_parent_map():
    snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snap == INVALID_HANDLE_VALUE:
        return {}
    out = {}
    pe = PROCESSENTRY32W()
    pe.dwSize = ctypes.sizeof(pe)
    try:
        ok = k32.Process32FirstW(snap, ctypes.byref(pe))
        while ok:
            out[int(pe.th32ProcessID)] = int(pe.th32ParentProcessID)
            ok = k32.Process32NextW(snap, ctypes.byref(pe))
    finally:
        k32.CloseHandle(snap)
    return out

def descendants(root):
    parents = process_parent_map()
    todo = [int(root)]
    seen = set()
    while todo:
        cur = todo.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for pid, ppid in list(parents.items()):
            if ppid == cur and pid not in seen:
                todo.append(pid)
    return seen

def threads_for_pid(pid):
    snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
    if snap == INVALID_HANDLE_VALUE:
        return []
    out = []
    te = THREADENTRY32()
    te.dwSize = ctypes.sizeof(te)
    try:
        ok = k32.Thread32First(snap, ctypes.byref(te))
        while ok:
            if int(te.th32OwnerProcessID) == int(pid):
                out.append(int(te.th32ThreadID))
            ok = k32.Thread32Next(snap, ctypes.byref(te))
    finally:
        k32.CloseHandle(snap)
    return out

def pin_pid(pid, slot, skip_ht_pair=True):
    try:
        n_groups = int(k32.GetActiveProcessorGroupCount())
        cpus_per = int(k32.GetActiveProcessorCount(0)) if n_groups else 64
        slots_per_group = max(1, cpus_per // (2 if skip_ht_pair else 1))
        group = int(slot // slots_per_group) % max(1, n_groups)
        cpu = int(slot % slots_per_group) * (2 if skip_ht_pair else 1)
        aff = GROUP_AFFINITY()
        aff.Mask = 1 << cpu
        aff.Group = group
        ok_any = False
        for tid in threads_for_pid(pid):
            h = k32.OpenThread(THREAD_SET_INFORMATION, False, tid)
            if not h:
                continue
            try:
                ok_any = bool(k32.SetThreadGroupAffinity(h, ctypes.byref(aff), None)) or ok_any
            finally:
                k32.CloseHandle(h)
        return ok_any, group, cpu
    except Exception:
        return False, -1, -1

def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--payload-file":
        with open(sys.argv[2], "rb") as f:
            payload_b64 = f.read()
    else:
        payload_b64 = sys.argv[1].encode("ascii")
    payload = json.loads(base64.b64decode(payload_b64).decode("utf-8"))
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in (payload.get("env") or {}).items()})
    cwd = payload["cwd"]
    log_path = payload["log_path"]
    pid_path = payload.get("wrapper_pid_path")
    if pid_path:
        try:
            os.makedirs(os.path.dirname(pid_path), exist_ok=True)
            with open(pid_path, "w", encoding="ascii") as f:
                f.write(str(os.getpid()))
        except Exception:
            pass
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "ab", buffering=0) as log:
        log.write(("scheduleurm windows wrapper start task=%s cwd=%s\n" % (payload.get("task_id"), cwd)).encode())
        cpu_plan = payload.get("cpu_plan") or {}
        if cpu_plan:
            log.write(("scheduleurm cpu-plan %s\n" % json.dumps(cpu_plan, sort_keys=True)).encode())
        if payload.get("argv"):
            proc = subprocess.Popen(payload["argv"], cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=CREATE_NEW_PROCESS_GROUP)
        else:
            proc = subprocess.Popen(payload["cmdline"], cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, shell=True, creationflags=CREATE_NEW_PROCESS_GROUP)
        log.write(("scheduleurm child pid=%s auto_pin=%s\n" % (proc.pid, payload.get("auto_pin"))).encode())
        assigned = {}
        next_slot = 0
        pin_base = int(payload.get("pin_base") or 0)
        pin_cores = max(1, int(payload.get("pin_cores") or 1))
        started_at = time.time()
        last_resource_log = 0.0
        loop_count = 0
        resource_interval = float(payload.get("resource_log_interval_s") or 60.0)
        while proc.poll() is None:
            loop_count += 1
            active = sorted(descendants(proc.pid))
            pin_ok_count = 0
            if payload.get("auto_pin"):
                assigned = {pid: slot for pid, slot in assigned.items() if pid in active}
                used_slots = set(assigned.values())
                for pid in active:
                    if pid not in assigned:
                        while next_slot in used_slots:
                            next_slot += 1
                        assigned[pid] = next_slot % pin_cores
                        used_slots.add(next_slot)
                        next_slot += 1
                    pin_slot = pin_base + (assigned[pid] % pin_cores)
                    ok, group, cpu = pin_pid(pid, pin_slot, bool(payload.get("skip_ht_pair", True)))
                    if ok:
                        pin_ok_count += 1
                    if ok and not os.environ.get("SCHEDULEURM_PIN_QUIET"):
                        log.write(("[pin] pid=%s slot=%s group=%s cpu=%s\n" % (pid, pin_slot, group, cpu)).encode())
            now = time.time()
            if now - last_resource_log >= resource_interval:
                progress = {
                    "task_id": payload.get("task_id"),
                    "elapsed_s": int(now - started_at),
                    "wrapper_loop": loop_count,
                    "root_pid": proc.pid,
                    "child_process_count": len(active),
                    "pinned_process_count": len(assigned),
                    "pin_ok_this_loop": pin_ok_count,
                    "next_pin_slot": next_slot,
                    "pin_base": pin_base,
                    "pin_cores": pin_cores,
                    "auto_pin": bool(payload.get("auto_pin")),
                    "cpu_plan": cpu_plan,
                }
                log.write(("scheduleurm resource-progress %s\n" % json.dumps(progress, sort_keys=True)).encode())
                last_resource_log = now
            time.sleep(float(payload.get("pin_interval_s", 1.0)))
        rc = proc.wait()
        log.write(("scheduleurm child exit rc=%s duration_s=%s assigned_processes=%s\n" % (
            rc, int(time.time() - started_at), len(assigned)
        )).encode())
    sys.exit(rc)

if __name__ == "__main__":
    main()
'''


_WINDOWS_DETACHER = r'''
import os, subprocess, sys

py, launcher, payload, outp, errp, pidp = sys.argv[1:7]
CREATE_NEW_PROCESS_GROUP = 0x00000200
DETACHED_PROCESS = 0x00000008
CREATE_BREAKAWAY_FROM_JOB = 0x01000000
flags = CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS | CREATE_BREAKAWAY_FROM_JOB
os.makedirs(os.path.dirname(outp), exist_ok=True)
os.makedirs(os.path.dirname(pidp), exist_ok=True)
out = open(outp, "ab", buffering=0)
err = open(errp, "ab", buffering=0)
p = subprocess.Popen(
    [py, launcher, "--payload-file", payload],
    stdin=subprocess.DEVNULL,
    stdout=out,
    stderr=err,
    close_fds=True,
    creationflags=flags,
)
with open(pidp, "w", encoding="ascii") as f:
    f.write(str(p.pid))
print("PID=" + str(p.pid))
'''


class WindowsBackend(Backend):
    """OpenSSH + PowerShell backend for Windows CPU-only nodes.

    jtl110cpu is a high-core-count Windows host. It cannot use the Linux
    LocalBackend assumptions (/proc, bash, setsid, nvidia-smi, rsync). This
    backend launches through a small Python wrapper on the Windows side. The
    wrapper starts the user's command, tracks descendants, and periodically
    pins worker processes across Windows processor groups to unique physical
    cores.
    """
    name = "windows"

    def _sched_dir(self, node: str) -> str:
        return (NODES.get(node, {}) or {}).get("windows_scheduleurm_dir") or r"F:\.scheduleurm"

    def _log_path(self, task: dict) -> str:
        return self._sched_dir(task["node"]).rstrip("\\/") + rf"\logs\{task['id']}.log"

    def _ensure_launcher(self, node: str) -> tuple[bool, str]:
        sched_dir = self._sched_dir(node)
        launcher_bytes = _WINDOWS_LAUNCHER.encode("utf-8")
        full_digest = hashlib.sha1(launcher_bytes).hexdigest()
        digest = full_digest[:12]
        # Use a stable content-hash wrapper path. If the file already exists, do
        # not hash or overwrite it while other wrappers may be running; the digest
        # in the filename is the version key. Fresh per-launch wrapper files were
        # observed to create short-lived Windows python processes with no main log
        # under heavy CPU fan-out, likely due immediate execution of newly-written
        # scripts over OpenSSH/PowerShell. Reusing an already-materialized wrapper
        # avoids that launch race and removes one SSH upload from every task.
        launcher = sched_dir.rstrip("\\/") + rf"\scheduleurm_win_wrapper_{digest}.py"
        ps = rf'''
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$dir={_ps_quote(sched_dir)}
$launcher={_ps_quote(launcher)}
$expected={_ps_quote(full_digest)}
[IO.Directory]::CreateDirectory($dir) | Out-Null
if (Test-Path -LiteralPath $launcher) {{
  Write-Output 'READY'
  exit 0
}}
$tmp = $launcher + ('.tmp.' + $PID)
$fs = [IO.File]::Open($tmp, [IO.FileMode]::Create, [IO.FileAccess]::Write, [IO.FileShare]::None)
try {{
  [Console]::OpenStandardInput().CopyTo($fs)
}} finally {{
  $fs.Close()
}}
$tmpHash = (Get-FileHash -Algorithm SHA1 -LiteralPath $tmp).Hash.ToLowerInvariant()
if ($tmpHash -ne $expected) {{
  Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
  throw ('launcher hash mismatch: ' + $tmpHash)
}}
if (Test-Path -LiteralPath $launcher) {{
  Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
  Write-Output 'READY'
  exit 0
}}
try {{
  Move-Item -LiteralPath $tmp -Destination $launcher
}} catch {{
  throw
}}
$actual = (Get-FileHash -Algorithm SHA1 -LiteralPath $launcher).Hash.ToLowerInvariant()
if ($actual -ne $expected) {{ throw ('launcher deploy verify failed: ' + $actual) }}
Write-Output 'READY'
'''
        encoded = base64.b64encode(ps.encode("utf-16le")).decode("ascii")
        try:
            proc = subprocess.run(
                _ssh_base_args(node) + [
                    "powershell",
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy", "Bypass",
                "-EncodedCommand", encoded,
                ],
                input=launcher_bytes,
                capture_output=True,
                timeout=20,
            )
            rc = proc.returncode
            out = (proc.stdout or b"").decode("utf-8", "replace")
            err = (proc.stderr or b"").decode("utf-8", "replace")
        except Exception as e:
            return False, str(e)[:200]
        if rc != 0 or "READY" not in out:
            return False, (err or out or f"launcher deploy rc={rc}").strip()[:200]
        return True, launcher

    def _ensure_detacher(self, node: str) -> tuple[bool, str]:
        sched_dir = self._sched_dir(node)
        detacher_bytes = _WINDOWS_DETACHER.encode("utf-8")
        full_digest = hashlib.sha1(detacher_bytes).hexdigest()
        digest = full_digest[:12]
        detacher = sched_dir.rstrip("\\/") + rf"\scheduleurm_win_detacher_{digest}.py"
        ps = rf'''
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$dir={_ps_quote(sched_dir)}
$detacher={_ps_quote(detacher)}
$expected={_ps_quote(full_digest)}
[IO.Directory]::CreateDirectory($dir) | Out-Null
if (Test-Path -LiteralPath $detacher) {{
  Write-Output 'READY'
  exit 0
}}
$tmp = $detacher + ('.tmp.' + $PID)
$fs = [IO.File]::Open($tmp, [IO.FileMode]::Create, [IO.FileAccess]::Write, [IO.FileShare]::None)
try {{
  [Console]::OpenStandardInput().CopyTo($fs)
}} finally {{
  $fs.Close()
}}
$tmpHash = (Get-FileHash -Algorithm SHA1 -LiteralPath $tmp).Hash.ToLowerInvariant()
if ($tmpHash -ne $expected) {{
  Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
  throw ('detacher hash mismatch: ' + $tmpHash)
}}
if (Test-Path -LiteralPath $detacher) {{
  Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
  Write-Output 'READY'
  exit 0
}}
Move-Item -LiteralPath $tmp -Destination $detacher
$actual = (Get-FileHash -Algorithm SHA1 -LiteralPath $detacher).Hash.ToLowerInvariant()
if ($actual -ne $expected) {{ throw ('detacher deploy verify failed: ' + $actual) }}
Write-Output 'READY'
'''
        encoded = base64.b64encode(ps.encode("utf-16le")).decode("ascii")
        try:
            proc = subprocess.run(
                _ssh_base_args(node) + [
                    "powershell",
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy", "Bypass",
                    "-EncodedCommand", encoded,
                ],
                input=detacher_bytes,
                capture_output=True,
                timeout=20,
            )
            rc = proc.returncode
            out = (proc.stdout or b"").decode("utf-8", "replace")
            err = (proc.stderr or b"").decode("utf-8", "replace")
        except Exception as e:
            return False, str(e)[:200]
        if rc != 0 or "READY" not in out:
            return False, (err or out or f"detacher deploy rc={rc}").strip()[:200]
        return True, detacher

    def launch(self, task: dict, node_state: Optional[dict] = None) -> tuple[bool, str]:
        node = task["node"]
        if int(task.get("est_vram_mb") or 0) > 0:
            block = _node_cpu_fallback_block_reason(task, node, NODES.get(node, {}))
            if block:
                return False, f"WindowsBackend is CPU-only; refusing GPU task: {block}"
            if not task.get("cpu_fallback_selected"):
                return False, "WindowsBackend is CPU-only; refusing GPU task"
        env_err = _windows_env_spec_error(task)
        if env_err:
            return False, env_err
        ok, launcher = self._ensure_launcher(node)
        if not ok:
            return False, f"windows launcher deploy failed: {launcher}"
        ok, detacher = self._ensure_detacher(node)
        if not ok:
            return False, f"windows detacher deploy failed: {detacher}"
        cwd = _windows_path_for_node(node, task.get("cwd") or "")
        log_path = self._log_path(task)
        py = (NODES.get(node, {}) or {}).get("windows_python") or "python"
        ps_cwd_check = f"if (Test-Path -LiteralPath {_ps_quote(cwd)}) {{ 'OK' }} else {{ 'MISSING' }}"
        try:
            rc_cwd, out_cwd, err_cwd = _run_windows_ps(node, ps_cwd_check, timeout=10, check=False)
        except Exception as e:
            return False, f"cwd check failed on {node}: {str(e)[:160]}"
        if rc_cwd != 0 or "OK" not in out_cwd:
            return False, f"cwd missing on {node}: {cwd}"

        cpu_plan = _apply_cpu_parallel_plan_to_task(task, node_state)
        env = {"CUDA_VISIBLE_DEVICES": "", "SCHEDULEURM_TASK_ID": task.get("id") or ""}
        # CPU worker-heavy defaults. User-supplied extra_env can still override
        # these below if a task has a specific threading model.
        env.update({
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "SCHEDULEURM_WINDOWS_AUTO_PIN": "1",
            "SCHEDULEURM_PIN_QUIET": "1",
        })
        cpu_env = _cpu_parallel_env(task) if cpu_plan else {}
        if cpu_env:
            env.update(cpu_env)
        for k, v in _safe_extra_env_items(_launch_extra_env(task)):
            if k == "CUDA_VISIBLE_DEVICES":
                continue
            env[k] = v
        cmd_payload = _windows_prepare_command(task)
        env.update(cmd_payload.pop("env", {}) or {})
        payload = {
            "task_id": task.get("id"),
            "cwd": cwd,
            "log_path": log_path,
            "env": env,
            "auto_pin": bool((NODES.get(node, {}) or {}).get("windows_auto_pin", True)),
            "skip_ht_pair": bool((NODES.get(node, {}) or {}).get("windows_skip_ht_pair", True)),
            "pin_base": int(task.get("windows_pin_base") or 0),
            "pin_cores": int(task.get("windows_pin_cores") or task.get("cpu_cores") or 1),
            "pin_interval_s": 1.0,
            "resource_log_interval_s": WINDOWS_WRAPPER_RESOURCE_LOG_INTERVAL_S,
            "cpu_plan": cpu_env,
        }
        payload.update(cmd_payload)
        sched_dir = self._sched_dir(node)
        payload_path = sched_dir.rstrip("\\/") + rf"\payloads\{task.get('id')}.b64"
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task.get("id") or "task"))
        launch_nonce = f"{time.time_ns()}_{os.getpid()}"
        pid_path = sched_dir.rstrip("\\/") + rf"\pids\{safe_id}_{launch_nonce}.pid"
        boot_out_path = log_path + f".{launch_nonce}.boot.out"
        boot_err_path = log_path + f".{launch_nonce}.boot.err"
        payload["wrapper_pid_path"] = pid_path
        payload_b64 = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
        write_payload_ps = rf'''
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$payloadPath={_ps_quote(payload_path)}
[IO.Directory]::CreateDirectory((Split-Path -Parent $payloadPath)) | Out-Null
$fs = [IO.File]::Open($payloadPath, [IO.FileMode]::Create, [IO.FileAccess]::Write, [IO.FileShare]::None)
try {{
  [Console]::OpenStandardInput().CopyTo($fs)
}} finally {{
  $fs.Close()
}}
Write-Output 'WROTE'
'''
        try:
            rc_payload, out_payload, err_payload = _run_windows_ps(
                node,
                write_payload_ps,
                timeout=20,
                check=False,
                input_data=payload_b64.encode("ascii"),
            )
        except Exception as e:
            return False, f"windows payload upload failed: {str(e)[:180]}"
        if rc_payload != 0 or "WROTE" not in out_payload:
            msg = (err_payload or out_payload or f"rc={rc_payload}").strip()
            return False, f"windows payload upload failed: {msg[:200]}"
        ps = rf'''
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$py={_ps_quote(py)}
$launcher={_ps_quote(launcher)}
$detacher={_ps_quote(detacher)}
$payloadPath={_ps_quote(payload_path)}
$logPath={_ps_quote(log_path)}
$pidPath={_ps_quote(pid_path)}
$bootOut={_ps_quote(boot_out_path)}
$bootErr={_ps_quote(boot_err_path)}
[IO.Directory]::CreateDirectory((Split-Path -Parent $logPath)) | Out-Null
[IO.Directory]::CreateDirectory((Split-Path -Parent $pidPath)) | Out-Null
Remove-Item -LiteralPath $bootOut,$bootErr -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
$detOut = & $py $detacher $py $launcher $payloadPath $bootOut $bootErr $pidPath
if ($LASTEXITCODE -ne 0) {{
  throw ('windows detacher failed: ' + (($detOut | Out-String).Trim()))
}}
Start-Sleep -Milliseconds 1800
$pidText = ''
if (Test-Path -LiteralPath $pidPath) {{
  $pidText = ([IO.File]::ReadAllText($pidPath)).Trim()
}}
if (-not $pidText) {{
  $pidText = (($detOut | Select-String -Pattern 'PID=(\d+)' | Select-Object -First 1).Matches.Groups[1].Value)
}}
$alive = $false
if ($pidText -match '^\d+$') {{
  $alive = [bool](Get-Process -Id ([int]$pidText) -ErrorAction SilentlyContinue)
}}
$hasLog = Test-Path -LiteralPath $logPath
if (-not $alive -and -not $hasLog) {{
  $boot = ''
  if (Test-Path -LiteralPath $bootErr) {{
    $boot = ((Get-Content -LiteralPath $bootErr -Tail 20 -ErrorAction SilentlyContinue) -join "`n")
  }}
  throw ('windows wrapper exited before writing task log; pid=' + $pidText + '; boot_err=' + $boot)
}}
Write-Output ('PID=' + $pidText)
exit 0
'''
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            local_ssh_log = LOG_DIR / f"{task.get('id')}.winssh.log"
            rc_launch, out_launch, err_launch = _run_windows_ps(node, ps, timeout=20, check=False)
            with open(local_ssh_log, "ab", buffering=0) as lf:
                if out_launch:
                    lf.write(out_launch.encode("utf-8", "replace"))
                if err_launch:
                    lf.write(err_launch.encode("utf-8", "replace"))
            if rc_launch != 0:
                msg = ((err_launch or "") + (out_launch or ""))
                return False, f"windows detached launch ssh rc={rc_launch}: {msg.strip()[:200]}"
            pid = None
            launch_out = out_launch or ""
            m = re.search(r"PID=(\d+)", launch_out)
            if m:
                pid = int(m.group(1))
            poll_ps = rf'''
$p={_ps_quote(pid_path)}
if (Test-Path -LiteralPath $p) {{ [IO.File]::ReadAllText($p).Trim() }}
'''
            if not pid:
                try:
                    rc_pid, out_pid, _ = _run_windows_ps(node, poll_ps, timeout=5, check=False)
                    s = (out_pid or "").strip()
                    if rc_pid == 0 and s.isdigit():
                        pid = int(s)
                except Exception:
                    pass
            if not pid:
                return False, f"windows detached launch did not produce wrapper pid; local_ssh_log={local_ssh_log}"
            task["remote_pids"] = [pid]
            task["process_group"] = pid
            task["log_path"] = log_path
            task["bootstrap_stdout_path"] = boot_out_path
            task["bootstrap_stderr_path"] = boot_err_path
            task.pop("local_ssh_pid", None)
            task["local_ssh_log_path"] = str(local_ssh_log)
            task["wrapper_pid_path"] = pid_path
            task["status"] = "running"
            task["started_at"] = time.time()
            _remember_last_placement(task)
            task["peak_vram_mb"] = 0
            task["peak_ram_mb"] = 0
            _set_current_usage(task, 0, 0, 0.0)
            return True, f"win_pid={pid}"
        except Exception as e:
            return False, f"windows launch exception: {str(e)[:200]}"

    def kill(self, task: dict, timeout: int = 15) -> tuple[bool, str]:
        node = task.get("node")
        pids = [int(p) for p in _task_pids(task) if p]
        if not node or not pids:
            return False, "no node/pids"
        roots = ",".join(str(p) for p in pids)
        ps = rf'''
$roots = @({roots})
$useCim = $true
try {{
  $procs = Get-CimInstance Win32_Process -ErrorAction Stop | Select-Object ProcessId,ParentProcessId
}} catch {{
  $useCim = $false
  $procs = @()
}}
$all = New-Object System.Collections.Generic.HashSet[int]
if ($useCim) {{
  $queue = New-Object System.Collections.Queue
  foreach ($r in $roots) {{ [void]$queue.Enqueue([int]$r) }}
  while ($queue.Count -gt 0) {{
    $cur = [int]$queue.Dequeue()
    if (-not $all.Add($cur)) {{ continue }}
    foreach ($p in $procs) {{
      if ([int]$p.ParentProcessId -eq $cur) {{ [void]$queue.Enqueue([int]$p.ProcessId) }}
    }}
  }}
}} else {{
  foreach ($r in $roots) {{ [void]$all.Add([int]$r) }}
}}
foreach ($procId in $all) {{ Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue }}
$deadline = (Get-Date).AddSeconds({max(1, int(timeout))})
do {{
  $alive = @()
  foreach ($procId in $all) {{
    $p = Get-Process -Id $procId -ErrorAction SilentlyContinue
    if ($p) {{ $alive += [int]$procId }}
  }}
  if ($alive.Count -eq 0) {{ break }}
  Start-Sleep -Milliseconds 250
}} while ((Get-Date) -lt $deadline)
if ($alive.Count -gt 0) {{
  Write-Output ("ALIVE_AFTER_KILL=" + (($alive | Sort-Object) -join ","))
  exit 45
}}
Write-Output ("STOPPED=" + (($all | Sort-Object) -join ","))
'''
        try:
            rc, out, err = _run_windows_ps(node, ps, timeout=timeout, check=False)
            if rc != 0:
                return False, (err or out or f"rc={rc}").strip()[:200]
            return True, out.strip()[:200]
        except Exception as e:
            return False, str(e)[:200]

    def batch_probe(self, state: dict) -> dict:
        by_node = {}
        for t in state["tasks"]:
            if t.get("status") != "running":
                continue
            node = t.get("node")
            if not node or not _node_is_windows(node):
                continue
            pids = _task_pids(t)
            if not pids:
                continue
            by_node.setdefault(node, []).append((t, pids))
        results = {}
        if not by_node:
            return results

        def _terminal_log_result(rc_int):
            return {
                "state": "dead",
                "alive_pids": [],
                "vram_mb": 0,
                "ram_mb": 0,
                "pcpu": 0.0,
                "terminal_ok": (rc_int == 0) if rc_int is not None else None,
                "backend_state": f"WINDOWS_LOG_RC_{rc_int}" if rc_int is not None else "WINDOWS_LOG_EXIT",
                "terminal_reason": (
                    f"windows wrapper child exit rc={rc_int}"
                    if rc_int is not None else
                    "windows wrapper child exit observed in log"
                ),
                "exit_code": rc_int,
            }

        def _probe_log_exits(node):
            """Cheap terminal fallback for saturated Windows CPU nodes."""
            specs = []
            # If the full process probe timed out, queue-accounting fallback can
            # otherwise keep dead Windows tasks "alive" indefinitely. Scan every
            # task log in that path; this is slower, but it is the correctness
            # fallback for saturated CPU nodes.
            scan_all = True
            for t, _pids in by_node[node]:
                log_path = t.get("log_path") or ""
                if not log_path:
                    continue
                known = []
                for x in (t.get("alive_pids") or []):
                    try:
                        known.append(int(x))
                    except Exception:
                        pass
                roots = []
                for x in (_pids or []):
                    try:
                        roots.append(int(x))
                    except Exception:
                        pass
                last_line = t.get("last_progress_line") or ""
                try:
                    progress_ratio = float(t.get("progress_ratio") or 0.0)
                except Exception:
                    progress_ratio = 0.0
                if progress_ratio <= 0:
                    try:
                        cur = float(t.get("runtime_current_unit") or 0.0)
                        total = float(t.get("runtime_total_units") or 0.0)
                        if total > 0:
                            progress_ratio = cur / total
                    except Exception:
                        pass
                try:
                    eta_seconds = float(t.get("eta_seconds") or 0.0)
                except Exception:
                    eta_seconds = 0.0
                if (not scan_all
                        and "scheduleurm child exit rc=" not in last_line
                        and not any(p in last_line for p in SUCCESS_PATTERNS)
                        and progress_ratio < 0.999
                        and not (eta_seconds > 0 and eta_seconds <= 600 and progress_ratio >= 0.98)):
                    continue
                specs.append({
                    "id": t.get("id"),
                    "log": _windows_path_for_task(t, log_path),
                    "roots": sorted(set(roots)),
                    "known": sorted(set(known)),
                })
            if not specs:
                return {}
            spec_json = json.dumps(specs)
            ps = rf'''
$specs = @(({_ps_quote(spec_json)} | ConvertFrom-Json))
$rows = @()
	foreach ($spec in $specs) {{
	  $id = [string]$spec.id
	  $log = [string]$spec.log
	  $all = New-Object System.Collections.Generic.HashSet[int]
	  foreach ($pid0 in @($spec.roots)) {{
	    try {{ [void]$all.Add([int]$pid0) }} catch {{ }}
	  }}
	  foreach ($pid0 in @($spec.known)) {{
	    try {{ [void]$all.Add([int]$pid0) }} catch {{ }}
	  }}
	  $logExit = $false
	  $logRc = $null
	  $logMtime = 0
  if ($log -and (Test-Path -LiteralPath $log)) {{
    try {{
      $it = Get-Item -LiteralPath $log -ErrorAction SilentlyContinue
      if ($it) {{ $logMtime = [int64]([DateTimeOffset]::new($it.LastWriteTimeUtc).ToUnixTimeSeconds()) }}
      $fs = [IO.File]::Open($log, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
      try {{
        $len = [int64]$fs.Length
        $start = [Math]::Max([int64]0, $len - 4096)
        [void]$fs.Seek($start, [IO.SeekOrigin]::Begin)
        $buf = New-Object byte[] ([int]($len - $start))
        $read = $fs.Read($buf, 0, $buf.Length)
        $text = [Text.Encoding]::UTF8.GetString($buf, 0, $read)
        foreach ($line in ($text -split "`r?`n")) {{
          if ($line -match 'scheduleurm windows wrapper start' -or
              $line -match 'scheduleurm child pid=' -or
              $line -match 'scheduleurm resource-progress') {{ $logExit = $false; $logRc = $null }}
	          if ($line -match 'scheduleurm child exit rc=([+-]?\d+)') {{
	            $logExit = $true
	            $logRc = [int]$Matches[1]
	          }}
	          if ($line -match 'scheduleurm child pid=(\d+)') {{ [void]$all.Add([int]$Matches[1]) }}
	          if ($line -match 'pid=(\d+)') {{ [void]$all.Add([int]$Matches[1]) }}
	          if ($line -match '"root_pid"\s*:\s*(\d+)') {{ [void]$all.Add([int]$Matches[1]) }}
	        }}
	      }} finally {{
	        $fs.Close()
	      }}
	    }} catch {{ }}
	  }}
	  $alive = @()
	  $rss = 0
	  foreach ($procId in $all) {{
	    try {{
	      $p = Get-Process -Id ([int]$procId) -ErrorAction SilentlyContinue
	      if (-not $p) {{ continue }}
	      $alive += [int]$procId
	      $rss += [int64]$p.WorkingSet64
	    }} catch {{ }}
	  }}
	  $rows += [pscustomobject]@{{
	    id=$id
	    alive=$alive
	    ram_mb=[int][math]::Round($rss / 1MB)
	    candidate_count=[int]$all.Count
	    log_exit=[bool]$logExit
	    exit_code=$logRc
	    log_mtime=[int64]$logMtime
  }}
}}
$rows | ConvertTo-Json -Compress -Depth 4
'''
            try:
                fallback_timeout = max(20, min(60, 8 + len(specs) // 2))
                rc, out, _ = _run_windows_ps(node, ps, timeout=fallback_timeout, check=False)
                if rc != 0 or not (out or "").strip():
                    return {}
                data = json.loads((out or "").strip())
                if isinstance(data, dict):
                    data = [data]
                if not isinstance(data, list):
                    return {}
                ret = {}
                for row in data:
                    tid = str(row.get("id") or "")
                    if not tid:
                        continue
                    rc_val = row.get("exit_code")
                    try:
                        rc_int = int(rc_val)
                    except Exception:
                        rc_int = None
                    if row.get("log_exit"):
                        ret[tid] = _terminal_log_result(rc_int)
                        continue
                    alive = row.get("alive") or []
                    if isinstance(alive, int):
                        alive = [alive]
                    alive = sorted(set(int(x) for x in alive))
                    if alive:
                        ret[tid] = {
                            "state": "alive",
                            "alive_pids": alive,
                            "vram_mb": 0,
                            "ram_mb": int(row.get("ram_mb") or 0),
                            "pcpu": 0.0,
                            "probe_fallback": "windows_log_pid_liveness",
                        }
                        continue
                    if int(row.get("candidate_count") or 0) > 0:
                        ret[tid] = {
                            "state": "dead",
                            "alive_pids": [],
                            "vram_mb": 0,
                            "ram_mb": 0,
                            "pcpu": 0.0,
                            "probe_fallback": "windows_log_pid_liveness",
                        }
                return ret
            except Exception:
                return {}

        def _probe(node):
            specs = []
            for t, pids in by_node[node]:
                known = []
                for x in (t.get("alive_pids") or []):
                    try:
                        known.append(int(x))
                    except Exception:
                        pass
                for p in pids:
                    specs.append({
                        "root": int(p),
                        "log": _windows_path_for_task(t, t.get("log_path") or ""),
                        "known": sorted(set(known)),
                        "started_at": float(t.get("started_at") or 0),
                        "finished_at": float(t.get("finished_at") or 0),
                    })
            roots = sorted({int(s["root"]) for s in specs})
            root_list = ",".join(str(p) for p in roots)
            spec_json = json.dumps(specs)
            ps = rf'''
$roots = @({root_list})
$specs = @(({_ps_quote(spec_json)} | ConvertFrom-Json))
$rows = @()
foreach ($spec in $specs) {{
  $r = [int]$spec.root
  $startedAt = [double]($spec.started_at)
  $finishedAt = [double]($spec.finished_at)
  $all = New-Object System.Collections.Generic.HashSet[int]
  [void]$all.Add([int]$r)
  foreach ($knownPid in @($spec.known)) {{
    try {{ [void]$all.Add([int]$knownPid) }} catch {{ }}
  }}
  $log = [string]$spec.log
  $logExit = $false
  $logRc = $null
  $logMtime = 0
  $seedCountBeforeLog = $all.Count
  if ($log -and (Test-Path -LiteralPath $log)) {{
    try {{
      $it = Get-Item -LiteralPath $log -ErrorAction SilentlyContinue
      if ($it) {{ $logMtime = [int64]([DateTimeOffset]::new($it.LastWriteTimeUtc).ToUnixTimeSeconds()) }}
    }} catch {{ }}
    $lines = Get-Content -LiteralPath $log -Tail 30 -ErrorAction SilentlyContinue
    foreach ($line in $lines) {{
      if ($line -match 'scheduleurm windows wrapper start' -or
          $line -match 'scheduleurm child pid=' -or
          $line -match 'scheduleurm resource-progress') {{ $logExit = $false; $logRc = $null }}
      if ($line -match 'scheduleurm child exit rc=([+-]?\d+)') {{ $logExit = $true; $logRc = [int]$Matches[1] }}
      if ($line -match 'scheduleurm child pid=(\d+)') {{ [void]$all.Add([int]$Matches[1]) }}
      if ($line -match 'pid=(\d+)') {{ [void]$all.Add([int]$Matches[1]) }}
      if ($line -match '"root_pid"\s*:\s*(\d+)') {{ [void]$all.Add([int]$Matches[1]) }}
    }}
  }}
  $alive = @()
  $rss = 0
  foreach ($procId in $all) {{
    try {{
      $p = Get-Process -Id ([int]$procId) -ErrorAction SilentlyContinue
      if (-not $p) {{ continue }}
      $keep = $true
      try {{
        if ($p.StartTime) {{
          $unix = [int64](([DateTimeOffset]$p.StartTime).ToUnixTimeSeconds())
          if ($startedAt -gt 0 -and $unix -lt ($startedAt - 120)) {{ $keep = $false }}
          if ($finishedAt -gt 0 -and $unix -gt ($finishedAt + 120)) {{ $keep = $false }}
        }}
      }} catch {{ }}
      if (-not $keep) {{ continue }}
      $alive += [int]$procId
      $rss += [int64]$p.WorkingSet64
    }} catch {{ }}
  }}
  $rows += [pscustomobject]@{{
    root=[int]$r
    alive=$alive
    ram_mb=[int][math]::Round($rss / 1MB)
    log_exit=[bool]$logExit
    exit_code=$logRc
    log_mtime=[int64]$logMtime
    log_seeded=[bool]($all.Count -gt $seedCountBeforeLog)
  }}
}}
$rows | ConvertTo-Json -Compress -Depth 4
'''
            try:
                probe_timeout = max(12, min(60, 6 + len(specs) // 2))
                rc, out, _ = _run_windows_ps(node, ps, timeout=probe_timeout, check=False)
                return out if rc == 0 else None
            except Exception:
                return None

        # Windows OpenSSH/PowerShell over the same public SSH gateway is fragile
        # when two large stdin-fed probe scripts run concurrently: one side can
        # stall in the temporary-script runner and the whole node reports
        # unknown/0 CPU. Probe Windows nodes sequentially; it is a display/control
        # loop, so correctness beats shaving a few seconds off refresh latency.
        high_fanout_threshold = max(1, int(os.environ.get(
            "SCHEDULEURM_WINDOWS_PROCESS_PROBE_HIGH_FANOUT", "32")))
        outputs = {}
        for node in by_node.keys():
            if len(by_node.get(node, [])) > high_fanout_threshold:
                # A single giant PowerShell probe over dozens of Windows tasks
                # consistently times out on saturated CPU nodes. Go straight to
                # the cheaper log/PID fallback, which still catches exits and
                # verifies recently logged child PIDs when possible.
                outputs[node] = None
            else:
                outputs[node] = _probe(node)
        log_exit_results = {
            node: _probe_log_exits(node)
            for node, out in outputs.items()
            if out is None
        }

        for node, out in outputs.items():
            if out is None:
                for t, pids in by_node[node]:
                    log_res = (log_exit_results.get(node) or {}).get(t["id"])
                    if log_res:
                        results[t["id"]] = log_res
                        continue
                    known = []
                    for x in (t.get("alive_pids") or []):
                        try:
                            known.append(int(x))
                        except Exception:
                            pass
                    known.extend(int(p) for p in pids if p)
                    ram = int(t.get("current_ram_mb") or t.get("peak_ram_mb") or t.get("ram_mb") or 0)
                    pcpu = float(max(1, int(t.get("cpu_cores") or 1)) * 100)
                    results[t["id"]] = {
                        "state": "alive",
                        "alive_pids": sorted(set(known)),
                        "vram_mb": 0,
                        "ram_mb": ram,
                        "pcpu": pcpu,
                        "probe_fallback": "windows_queue_accounting",
                    }
                continue
            try:
                raw = (out or "").strip()
                if not raw:
                    raise ValueError("empty windows process probe output")
                data = json.loads(raw)
                if isinstance(data, dict):
                    data = [data]
                if not isinstance(data, list) or not data:
                    raise ValueError("windows process probe returned no rows")
            except Exception as e:
                log_res_by_id = _probe_log_exits(node)
                for t, pids in by_node[node]:
                    log_res = log_res_by_id.get(t["id"])
                    if log_res:
                        results[t["id"]] = log_res
                        continue
                    known = []
                    for x in (t.get("alive_pids") or []):
                        try:
                            known.append(int(x))
                        except Exception:
                            pass
                    known.extend(int(p) for p in pids if p)
                    ram = int(t.get("current_ram_mb") or t.get("peak_ram_mb") or t.get("ram_mb") or 0)
                    pcpu = float(max(1, int(t.get("cpu_cores") or 1)) * 100)
                    results[t["id"]] = {
                        "state": "alive",
                        "alive_pids": sorted(set(known)),
                        "vram_mb": 0,
                        "ram_mb": ram,
                        "pcpu": pcpu,
                        "probe_fallback": f"windows_parse_failed: {str(e)[:80]}",
                    }
                continue
            by_root = {int(r.get("root")): r for r in data if r.get("root") is not None}
            for t, pids in by_node[node]:
                alive = []
                ram = 0
                pcpu = 0.0
                for p in pids:
                    rec = by_root.get(int(p))
                    if not rec:
                        continue
                    if rec.get("log_exit"):
                        rc_val = rec.get("exit_code")
                        try:
                            rc_int = int(rc_val)
                        except Exception:
                            rc_int = None
                        results[t["id"]] = _terminal_log_result(rc_int)
                        break
                    a = rec.get("alive") or []
                    if isinstance(a, int):
                        a = [a]
                    alive.extend(int(x) for x in a)
                    ram += int(rec.get("ram_mb") or 0)
                    try:
                        pcpu += float(rec.get("pcpu") or 0.0)
                    except Exception:
                        pass
                if t["id"] in results:
                    continue
                if not alive:
                    if _local_launch_transport_alive(t):
                        results[t["id"]] = {"state": "unknown", "alive_pids": [],
                                            "vram_mb": 0, "ram_mb": 0, "pcpu": 0.0,
                                            "error": "windows root missing but local launch transport is alive"}
                        continue
                    results[t["id"]] = {"state": "dead", "alive_pids": [],
                                        "vram_mb": 0, "ram_mb": 0, "pcpu": 0.0}
                else:
                    if pcpu <= 0:
                        pcpu = float(max(1, int(t.get("cpu_cores") or 1)) * 100)
                    results[t["id"]] = {"state": "alive", "alive_pids": sorted(set(alive)),
                                        "vram_mb": 0, "ram_mb": ram, "pcpu": pcpu}
        return results


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
        remote_cwd = _remote_path_for_node(task["node"], cwd)
        inner = _apply_node_cmd_rewrites(task.get("node"), task["cmd"])
        inner = _rewrite_command_paths_for_node(task.get("node"), inner)
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
            inner = f"{inner} {resume_flag} {shlex.quote(_remote_path_for_node(task['node'], resume_path))}"
        # Env-deploy wrapping (docker). Done AFTER -u + resume injection so the inner shell cmd
        # is fully assembled before being wrapped in `docker run`. For env_spec='none' this is a
        # no-op; for 'auto', probes target and falls back to 'none' if docker isn't accessible;
        # for explicit 'docker' returns an error that fails the launch fast (per Codex review,
        # silent host fallback was unsafe).
        inner, docker_err = _maybe_wrap_docker(task, inner, remote_cwd)
        if docker_err:
            return False, docker_err
        # Pre-flight: cwd must exist on target node. Skips a wasted launch + 2-3 retry cycles +
        # eventual ENV_MISSING escalation when a node simply doesn't have the repo synced.
        try:
            rc_cwd, _, err_cwd = run_on(task["node"], f"test -d {shlex.quote(remote_cwd)}", timeout=10, check=False)
        except Exception as e:
            rc_cwd, err_cwd = 1, str(e)
        if rc_cwd != 0:
            return False, f"cwd missing on {task['node']}: {remote_cwd}"
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
        for k, v in _safe_extra_env_items(_node_launch_extra_env(task)):
            env_prefix += f"export {k}={shlex.quote(v)}; "
        # setsid creates a new session + process group leader, so `cancel --force`'s `kill -- -<pid>`
        # reliably catches every worker child. The </dev/null is so the launched process doesn't
        # inherit ssh's stdin pipe (which would otherwise keep the ssh-side bash alive).
        full = (f"mkdir -p {shlex.quote(os.path.dirname(log_path))}; "
                f"cd {shlex.quote(remote_cwd)} && {env_prefix} "
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
                                        "vram_mb": 0, "ram_mb": 0, "pcpu": 0.0,
                                        "error": "local backend ssh/proc probe failed"}
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

    def requires_local_capacity_check(self, node: str, task: Optional[dict] = None,
                                      node_state: Optional[dict] = None) -> bool:
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

    def _build_sbatch_script(self, task: dict, inner_cmd: str, log_path: str,
                             cwd: Optional[str] = None) -> str:
        """Build the sbatch script as a string. Streamed via stdin to `sbatch /dev/stdin`."""
        script_cwd = cwd or task["cwd"]
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
        node_info = NODES.get(task.get("node"), {}) or {}
        default_partition = (
            node_info.get("slurm_gpu_partition")
            if vram > 0 else node_info.get("slurm_cpu_partition")
        )
        partition = task.get("slurm_partition") or default_partition
        if partition:
            lines.append(f"#SBATCH --partition={partition}")
        # Optional slurm-specific fields if user set them on the task
        for slurm_field, sbatch_flag in (
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
        for k, v in _safe_extra_env_items(_node_launch_extra_env(task)):
            lines.append(f"export {k}={shlex.quote(v)}")
        # Phase 2.5 P1 fix: cd must be fatal-on-failure. A bare `cd path` followed by
        # the inner cmd silently continues from $HOME (or wherever) if the compute node
        # doesn't see this path (NFS stale handle, cwd not propagated to compute, etc.).
        # The job appears to "run" but produces no output, hits no checkpoints, and
        # diagnose has no signal — log just has whatever bash printed about cd. Match
        # LocalBackend's `cd ... && cmd` semantics with an explicit guard that also
        # leaves a parseable error in the log so diagnose can route to ENV_MISSING.
        cwd_q = shlex.quote(script_cwd)
        lines.append(
            f"cd {cwd_q} || {{ "
            f"echo \"scheduleurm: cwd not accessible on compute node: {script_cwd}\" >&2; "
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
        remote_cwd = _remote_path_for_node(task["node"], cwd)
        log_path = (f"{STATE_DIR}/logs/{task['id']}.log"
                    if NODES[task["node"]]["host"] is None
                    else f"{remote_cwd}/.scheduleurm/{task['id']}.log")
        inner = _apply_node_cmd_rewrites(task.get("node"), task["cmd"])
        inner = _rewrite_command_paths_for_node(task.get("node"), inner)
        inner = _inject_python_u(inner)
        resume_path = task.get("resume_from")
        resume_flag = task.get("resume_flag") or ""
        if resume_path and resume_flag:
            inner = f"{inner} {resume_flag} {shlex.quote(_remote_path_for_node(task['node'], resume_path))}"
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
        inner, docker_err = _maybe_wrap_docker(task, inner, remote_cwd, gpu_runtime_env=gpu_runtime_env)
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
                f"test -d {shlex.quote(remote_cwd)} && mkdir -p {shlex.quote(log_dir)}",
                timeout=10, check=False,
            )
        except Exception as e:
            rc_cwd = 1
        if rc_cwd != 0:
            return False, f"cwd missing or log_dir uncreatable on {task['node']}: {remote_cwd}"

        script = self._build_sbatch_script(task, inner, log_path, cwd=remote_cwd)
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
                # Deliberately do not use _ssh_no_stdin_args() here: sbatch reads
                # the script from stdin via /dev/stdin.
                proc = subprocess.run(
                    _ssh_base_args(task["node"]) + ["sbatch /dev/stdin"],
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
                                        "vram_mb": 0, "ram_mb": 0, "pcpu": 0.0,
                                        "error": "slurm squeue probe failed"}
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


def _slurm_mode_disabled(value) -> bool:
    return str(value or "").strip().lower() in ("local", "false", "0", "no", "off", "none")


def _as_int_or_none(value) -> Optional[int]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, (list, tuple, set)):
        return len([v for v in value if str(v).strip()])
    text = str(value).strip()
    if not text:
        return None
    if "," in text:
        parts = [p.strip() for p in text.split(",") if p.strip() and p.strip() not in ("-1", "none", "None")]
        if parts:
            return len(parts)
    if ":" in text:
        tail = text.rsplit(":", 1)[-1].strip()
        if tail.isdigit():
            return int(tail)
    m = re.search(r"(\d+)", text)
    if not m:
        return None
    return int(m.group(1))


def _node_declared_gpu_count(info: dict, node_state: Optional[dict] = None) -> int:
    """Best-effort GPU count for Slurm auto policy.

    NODES is authoritative when it declares a count; probe state fills in the
    common case for directly reachable GPU boxes. Slurm login/controller nodes
    may probe as gpus=[], so CPU scale is checked separately.
    """
    sources = [info or {}]
    if node_state:
        sources.append(node_state)
    for src in sources:
        for key in ("slurm_auto_gpu_count", "gpu_count", "num_gpus", "total_gpus", "ngpus"):
            val = _as_int_or_none(src.get(key))
            if val is not None:
                return max(0, val)
        gpus = src.get("gpus")
        if isinstance(gpus, (list, tuple, set)):
            return len(gpus)
        val = _as_int_or_none(gpus)
        if val is not None:
            return max(0, val)
    return 0


def _node_is_large_slurm_candidate(node: str, node_state: Optional[dict] = None) -> bool:
    """Whether this node looks like a shared/cluster-class Slurm target."""
    info = NODES.get(node, {}) or {}
    override = info.get("slurm_auto_large")
    if override is not None:
        return str(override).strip().lower() in ("true", "1", "yes", "on", "large")
    cpu_threshold = int(info.get("slurm_auto_min_cpu_cores") or SLURM_AUTO_MIN_CPU_CORES)
    gpu_threshold = int(info.get("slurm_auto_min_gpus") or SLURM_AUTO_MIN_GPUS)
    cpu_cores = _as_int_or_none(info.get("slurm_auto_cpu_cores"))
    if cpu_cores is None:
        cpu_cores = _as_int_or_none(info.get("cpu_cores"))
    if cpu_cores is None and node_state:
        cpu_cores = _as_int_or_none(node_state.get("cpu_cores") or node_state.get("total_cpu"))
    gpu_count = _node_declared_gpu_count(info, node_state=node_state)
    return bool((cpu_cores or 0) >= cpu_threshold or gpu_count >= gpu_threshold)


def _task_requested_gpu_count(task: Optional[dict]) -> int:
    if not task:
        return 0
    for key in ("gpu_count", "num_gpus", "n_gpus", "ngpus", "gpus",
                "slurm_gpus", "nproc_per_node", "num_processes"):
        val = _as_int_or_none(task.get(key))
        if val is not None:
            return max(0, val)
    cmd = str(task.get("cmd") or task.get("command") or "")
    if not cmd:
        return 0
    env_m = re.search(r"(?:^|\s)CUDA_VISIBLE_DEVICES=([^\s]+)", cmd)
    if env_m:
        val = _as_int_or_none(env_m.group(1))
        if val is not None:
            return max(0, val)
    gres_m = re.search(r"--gres(?:=|\s+)gpu(?::[A-Za-z0-9_.-]+)?:(\d+)", cmd)
    if gres_m:
        return int(gres_m.group(1))
    flags = {
        "--nproc_per_node", "--nproc-per-node",
        "--num_processes", "--num-processes",
        "--num_gpus", "--num-gpus",
        "--gpus", "--n_gpu", "--n-gpu", "--ngpus", "--nproc",
        "--gpus-per-node", "--gpus_per_node",
    }
    try:
        tokens = shlex.split(cmd)
    except Exception:
        tokens = cmd.split()
    for i, tok in enumerate(tokens):
        for flag in flags:
            if tok == flag and i + 1 < len(tokens):
                val = _as_int_or_none(tokens[i + 1])
                if val is not None:
                    return max(0, val)
            if tok.startswith(flag + "="):
                val = _as_int_or_none(tok.split("=", 1)[1])
                if val is not None:
                    return max(0, val)
    return 0


def _task_is_llm_like(task: Optional[dict]) -> bool:
    if not task:
        return False
    haystack = " ".join(
        str(task.get(k) or "")
        for k in ("project", "signature", "description", "cmd", "command", "cwd")
    ).lower()
    keywords = (
        "llm", "large language", "language_model", "language-model",
        "fine-tune", "fine_tune", "finetune", "fine tuning",
        "sft", "lora", "qlora", "peft", "trl", "deepspeed",
        "accelerate launch", "torchrun", "transformers", "huggingface",
        "llama", "qwen", "mistral", "chatglm", "baichuan", "yi-",
    )
    return any(k in haystack for k in keywords)


def _task_prefers_slurm_auto(task: Optional[dict]) -> bool:
    """Large task heuristic for hardware-aware Slurm auto routing.

    Small one-GPU jobs deliberately stay on LocalBackend so scheduleurm can pack
    several per GPU when VRAM/CPU/RAM allow it.
    """
    if not task:
        return False
    if _task_requested_gpu_count(task) >= SLURM_AUTO_LARGE_TASK_MIN_GPUS:
        return True
    if _task_is_llm_like(task):
        return True
    if int((task or {}).get("est_vram_mb") or 0) >= SLURM_AUTO_LARGE_TASK_VRAM_MB:
        return True
    if int((task or {}).get("cpu_cores") or 0) >= SLURM_AUTO_LARGE_TASK_CPU_CORES:
        return True
    return False


class HybridBackend(Backend):
    """Per-node routing: default LocalBackend; SlurmBackend is opt-in/auto.

    Slurm detection is still a capability probe, not a blanket policy. Small
    personal nodes with Slurm installed keep scheduleurm's VRAM/RAM/CPU packing;
    large/shared nodes can auto-route LLM/multi-GPU/heavy jobs to Slurm.
    """
    name = "hybrid"

    def __init__(self):
        self._local = LocalBackend()
        self._slurm = SlurmBackend()
        self._windows = WindowsBackend()
        self._cache: dict = {}  # node_name -> 'slurm' | 'local'

    def _node_wants_slurm(self, node: str, task: Optional[dict] = None,
                          node_state: Optional[dict] = None) -> bool:
        """Policy decision: should this future launch use SlurmBackend?

        Defaults are deliberately local for small jobs. Operators can force Slurm
        globally per node (`slurm_backend="slurm"`), per resource bucket
        (`slurm_gpu_backend="slurm"` / `slurm_cpu_backend="slurm"`), force local
        with `local`/`false`, or use explicit task Slurm fields. `auto` and the
        default hardware-aware path only choose Slurm for large/shared nodes plus
        jobs that look large enough for Slurm to be the right owner.
        """
        if _node_is_windows(node):
            return False
        info = NODES.get(node, {}) or {}
        slurm_backend = info.get("slurm_backend")
        if _slurm_mode_disabled(slurm_backend):
            return False
        if _slurm_mode_enabled(info.get("slurm_backend")):
            return True

        is_gpu_task = int((task or {}).get("est_vram_mb", DEFAULT_VRAM_MB) or 0) > 0
        bucket_key = "slurm_gpu_backend" if is_gpu_task else "slurm_cpu_backend"
        bucket_backend = info.get(bucket_key)
        if _slurm_mode_disabled(bucket_backend):
            return False
        if _slurm_mode_enabled(bucket_backend):
            return True

        # Explicit task Slurm knobs mean "send me to a Slurm-capable node", but
        # keep non-Slurm local nodes out of the candidate set in pick_placement.
        if _task_requests_slurm(task):
            return self._kind_for(node) == "slurm"

        auto_candidate = _node_is_large_slurm_candidate(node, node_state=node_state)
        heavy_task = _task_prefers_slurm_auto(task)
        # `auto` now shares the default hardware-aware policy instead of the old
        # "Slurm installed => always Slurm" behavior.
        if auto_candidate and heavy_task:
            return self._kind_for(node) == "slurm"
        return False

    def requires_local_capacity_check(self, node: str, task: Optional[dict] = None,
                                      node_state: Optional[dict] = None) -> bool:
        """True for scheduleurm-managed placement; False for Slurm-routed nodes."""
        return not self._node_wants_slurm(node, task, node_state=node_state)

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

    def _backend_for(self, node: str, task: Optional[dict] = None,
                     node_state: Optional[dict] = None) -> Backend:
        if _node_is_windows(node):
            return self._windows
        return self._slurm if self._node_wants_slurm(node, task, node_state=node_state) else self._local

    def _backend_for_task(self, task: dict, node_state: Optional[dict] = None) -> Backend:
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
        - neither → queued, use current Slurm routing policy for the upcoming launch
        """
        node = task.get("node")
        if node and _node_is_windows(node):
            return self._windows
        if task.get("slurm_job_id"):
            return self._slurm
        if task.get("remote_pids"):
            return self._local
        if not node:
            return self._local  # no node yet (queued task): launch path will re-route
        return self._backend_for(node, task, node_state=node_state)

    def launch(self, task: dict, node_state: Optional[dict] = None) -> tuple[bool, str]:
        return self._backend_for_task(task, node_state=node_state).launch(task, node_state=node_state)

    def kill(self, task: dict, timeout: int = 15) -> tuple[bool, str]:
        return self._backend_for_task(task).kill(task, timeout=timeout)

    def batch_probe(self, state: dict) -> dict:
        """Split tasks per backend, probe each, merge results. Two ssh round-trips per node
        in the worst case (one local probe + one slurm probe), but each backend's batch_probe
        skips nodes with no relevant tasks so common case is one round-trip per node."""
        # Synthesize per-backend state subsets so each backend only sees its own tasks.
        local_tasks, slurm_tasks, windows_tasks = [], [], []
        for t in state["tasks"]:
            if t["status"] != "running":
                continue
            backend = self._backend_for_task(t)
            if backend is self._slurm:
                slurm_tasks.append(t)
            elif backend is self._windows:
                windows_tasks.append(t)
            else:
                local_tasks.append(t)
        merged: dict = {}
        if local_tasks:
            merged.update(self._local.batch_probe({"tasks": local_tasks}))
        if slurm_tasks:
            merged.update(self._slurm.batch_probe({"tasks": slurm_tasks}))
        if windows_tasks:
            merged.update(self._windows.batch_probe({"tasks": windows_tasks}))
        return merged


# Singleton: HybridBackend defaults small jobs to scheduleurm LocalBackend placement;
# SlurmBackend is used for explicit task Slurm fields, forced config, or
# hardware-aware large-task auto routing on large Slurm-capable nodes.
# Tests reference _BACKEND directly to verify backend identity and to swap in fakes.
_BACKEND: Backend = HybridBackend()


def _requires_local_capacity_check(node: str, task: Optional[dict] = None,
                                   node_state: Optional[dict] = None) -> bool:
    """Compatibility wrapper for tests/plugins with older backend stubs."""
    try:
        return _BACKEND.requires_local_capacity_check(node, task=task, node_state=node_state)
    except TypeError:
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

def _seed_value_substitute(value, seed: str):
    """Substitute the simple shell seed forms used by BAPR batch submit commands."""
    if value is None:
        return None
    if isinstance(value, list):
        return [_seed_value_substitute(v, seed) for v in value]
    text = str(value)
    text = text.replace("${seed}", str(seed))
    text = re.sub(r"\$seed\b", str(seed), text)
    return text

def _submit_context_is_bapr(cmd: str = "", cwd: str = "", signature: str = "",
                            project: str = "") -> bool:
    hay = " ".join(str(x or "") for x in (cmd, cwd, signature, project))
    return (
        str(signature or "").startswith("BAPR/")
        or str(project or "") == "BAPR"
        or "/BAPR" in hay
        or "/bapr_v15" in hay
    )

def _expand_simple_seed_loop_inner(inner: str):
    """Expand `prefix; for seed in 0 1; do body; done` into per-seed shell bodies.

    This is deliberately conservative: it only handles literal integer seed lists.
    More complex shell loops stay as-is unless the caller rewrites them explicitly.
    """
    if "for seed in" not in (inner or ""):
        return []
    m = re.match(
        r"(?s)^(?P<prefix>.*?)\bfor\s+seed\s+in\s+"
        r"(?P<seeds>[0-9][0-9 \t]*)\s*;\s*do\s+"
        r"(?P<body>.*?)\s*;\s*done\s*$",
        inner.strip(),
    )
    if not m:
        return []
    seeds = [s for s in m.group("seeds").split() if s]
    if len(seeds) <= 1:
        return []
    prefix = m.group("prefix").strip()
    while prefix.endswith(";"):
        prefix = prefix[:-1].rstrip()
    body = m.group("body").strip()
    expanded = []
    for seed in seeds:
        seed_body = _seed_value_substitute(body, seed)
        new_inner = "; ".join([p for p in (prefix, seed_body) if p])
        expanded.append((seed, new_inner))
    return expanded

def _split_bapr_seed_batch_submit_args(args):
    """Return per-seed submit args for BAPR seed-loop commands, else [].

    The scheduler stays generic, but BAPR training/eval batches have repeatedly
    hidden multiple long seed runs inside one task. For BAPR only, split those
    simple shell loops at submit time so resource placement and requeue happen
    per seed.
    """
    if getattr(args, "allow_seed_batch", False):
        return []
    if not _submit_context_is_bapr(
        getattr(args, "cmd", ""),
        getattr(args, "cwd", ""),
        getattr(args, "signature", ""),
        getattr(args, "project", ""),
    ):
        return []
    try:
        toks = shlex.split(getattr(args, "cmd", "") or "")
    except Exception:
        return []
    if len(toks) < 3 or os.path.basename(toks[0]) not in ("bash", "sh", "zsh", "dash"):
        return []
    c_i = None
    for i, tok in enumerate(toks[1:], 1):
        if tok in ("-c", "-lc"):
            c_i = i
            break
    if c_i is None or c_i + 1 >= len(toks):
        return []
    expanded_inners = _expand_simple_seed_loop_inner(toks[c_i + 1])
    if not expanded_inners:
        return []
    result = []
    for seed, new_inner in expanded_inners:
        vals = vars(args).copy()
        new_toks = list(toks)
        new_toks[c_i + 1] = new_inner
        vals["cmd"] = " ".join(shlex.quote(t) for t in new_toks)
        sig = str(getattr(args, "signature", "") or "").rstrip("/")
        vals["signature"] = f"{sig}/s{seed}" if sig else f"s{seed}"
        desc = str(getattr(args, "description", "") or "").strip()
        vals["description"] = f"{desc} seed {seed}".strip()
        for key in ("ckpt_dir", "result_dir", "local_result_dir"):
            if key in vals and vals[key]:
                vals[key] = _seed_value_substitute(vals[key], seed)
        if vals.get("wait_for_files"):
            vals["wait_for_files"] = _seed_value_substitute(vals["wait_for_files"], seed)
        result.append(argparse.Namespace(**vals))
    return result

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
        "result_dir": os.path.join(_abs_under(script_cwd, save_root), run_name),
        "ckpt_dir": os.path.join(_abs_under(script_cwd, save_root), run_name, "checkpoints"),
        "resume_managed_by_cmd": _cmd_has_resume_flag(text),
        "source": f"wrapper:{os.path.basename(script)}",
    }

def _infer_bapr_run_seed_ckpt(cmd: str, cwd: str = ""):
    """Infer BAPR run_seed.sh checkpoints even when hidden inside `bash -lc`.

    `run_seed.sh` computes:
      RUN_NAME="${TAG}_${ALGO}_${ENV%-v2}_dw${DWELL}_s${SEED}"
      save_root=jax_experiments/results_paper
    so the checkpoint path is deterministic from the wrapper args.
    """
    candidates = []
    try:
        toks = shlex.split(cmd or "")
    except Exception:
        toks = []
    if toks:
        candidates.append(toks)
        for i, tok in enumerate(toks[:-1]):
            if tok in ("-c", "-lc"):
                try:
                    candidates.append(shlex.split(toks[i + 1]))
                except Exception:
                    pass
    for ctoks in candidates:
        for i, tok in enumerate(ctoks):
            if not tok.endswith("run_seed.sh"):
                continue
            args = ctoks[i + 1:]
            if args and os.path.basename(tok) == "run_seed.sh":
                script_path = tok
            else:
                continue
            if len(args) < 3:
                continue
            algo, env_name, seed = args[0], args[1], args[2]
            dwell = args[4] if len(args) >= 5 and args[4] else "60"
            tag = args[6] if len(args) >= 7 and args[6] else "paper"
            if any(("$" in str(x) or ";" in str(x)) for x in (algo, env_name, seed, dwell, tag)):
                continue
            env_short = env_name[:-3] if env_name.endswith("-v2") else env_name
            run_name = f"{tag}_{algo}_{env_short}_dw{dwell}_s{seed}"
            if os.path.isabs(script_path):
                base = os.path.dirname(script_path)
            else:
                base = cwd or os.getcwd()
            result_dir = os.path.join(base, "jax_experiments", "results_paper", run_name)
            return {
                "result_dir": result_dir,
                "ckpt_dir": os.path.join(result_dir, "checkpoints"),
                "resume_managed_by_cmd": True,
                "source": "wrapper:run_seed.sh",
            }
    return None

def _infer_bapr_result_dirs_from_cmd(cmd: str, cwd: str = "") -> list:
    """Infer one or more BAPR run directories from run_seed.sh commands."""
    out = []
    inferred = _infer_bapr_run_seed_ckpt(cmd, cwd)
    if inferred and inferred.get("result_dir"):
        out.append(inferred["result_dir"])

    try:
        toks = shlex.split(cmd or "")
    except Exception:
        toks = []
    for i, tok in enumerate(toks[:-1]):
        if tok not in ("-c", "-lc"):
            continue
        for _, inner in _expand_simple_seed_loop_inner(toks[i + 1]):
            shell = " ".join([shlex.quote(toks[0]), shlex.quote(tok), shlex.quote(inner)])
            inferred = _infer_bapr_run_seed_ckpt(shell, cwd)
            if inferred and inferred.get("result_dir"):
                out.append(inferred["result_dir"])

    dedup = []
    seen = set()
    for path in out:
        key = _conflict_path_key(path)
        if not key or key in seen:
            continue
        seen.add(key)
        dedup.append(path)
    return dedup

def _bapr_run_seed_metas_from_cmd(cmd: str, cwd: str = "") -> list[dict]:
    """Return per-seed metadata for BAPR `run_seed.sh` invocations in a command."""
    try:
        toks = shlex.split(cmd or "")
    except Exception:
        toks = []
    candidate_token_lists = []
    if toks:
        candidate_token_lists.append(toks)
        for i, tok in enumerate(toks[:-1]):
            if tok in ("-c", "-lc"):
                inner = toks[i + 1]
                try:
                    candidate_token_lists.append(shlex.split(inner))
                except Exception:
                    pass
                for _seed, expanded_inner in _expand_simple_seed_loop_inner(inner):
                    try:
                        candidate_token_lists.append(shlex.split(expanded_inner))
                    except Exception:
                        pass

    metas = []
    seen = set()
    for ctoks in candidate_token_lists:
        for i, tok in enumerate(ctoks):
            if not tok.endswith("run_seed.sh"):
                continue
            args = ctoks[i + 1:]
            if os.path.basename(tok) != "run_seed.sh" or len(args) < 3:
                continue
            algo, env_name, seed = args[0], args[1], args[2]
            max_iters = args[3] if len(args) >= 4 and args[3] else "1500"
            dwell = args[4] if len(args) >= 5 and args[4] else "60"
            target_mode = args[5] if len(args) >= 6 and args[5] else "min"
            tag = args[6] if len(args) >= 7 and args[6] else "paper"
            if any(("$" in str(x) or ";" in str(x)) for x in (
                algo, env_name, seed, max_iters, dwell, target_mode, tag,
            )):
                continue
            try:
                max_iters_i = int(max_iters)
            except Exception:
                continue
            env_short = env_name[:-3] if env_name.endswith("-v2") else env_name
            run_name = f"{tag}_{algo}_{env_short}_dw{dwell}_s{seed}"
            base = os.path.dirname(tok) if os.path.isabs(tok) else (cwd or os.getcwd())
            result_dir = os.path.join(base, "jax_experiments", "results_paper", run_name)
            key = _conflict_path_key(result_dir)
            if not key or key in seen:
                continue
            seen.add(key)
            metas.append({
                "algo": algo,
                "env": env_name,
                "seed": str(seed),
                "max_iters": max_iters_i,
                "dwell": str(dwell),
                "target_mode": str(target_mode),
                "tag": str(tag),
                "run_name": run_name,
                "result_dir": result_dir,
            })
    return metas

def _bapr_batch_projection(task: dict, tail_text: str, elapsed_s: float) -> Optional[dict]:
    """Whole-task ETA for legacy BAPR multi-seed batches.

    BAPR logs print `Iter N` for the current seed, while the old batch command
    hides total work in `for seed ... run_seed.sh ... 1500 ...`. Generic ETA
    can only see the current seed or history. This projects across all seeds.
    """
    metas = _bapr_run_seed_metas_from_cmd(task.get("cmd", ""), task.get("cwd", ""))
    if len(metas) <= 1:
        return None
    et = _load_eta_tracker_module()
    if not et:
        return None
    try:
        current = et._extract_current_only_from_tail(tail_text)
    except Exception:
        current = None
    if current is None:
        try:
            progress = et.parse_progress(tail_text, cmd=task.get("cmd"))
            current = progress[0] if progress else None
        except Exception:
            current = None

    run_to_idx = {m["run_name"]: i for i, m in enumerate(metas)}
    completed = set()
    for m in re.finditer(r"Results saved to:\s*(\S+)", tail_text or ""):
        raw = _clean_result_path(m.group(1))
        parts = Path(raw).parts
        run_name = ""
        if parts:
            run_name = parts[-2] if parts[-1] == "logs" and len(parts) >= 2 else parts[-1]
        idx = run_to_idx.get(run_name)
        if idx is not None:
            completed.add(idx)
    for i, meta in enumerate(metas):
        try:
            if (Path(meta["result_dir"]) / "done.ok").exists():
                completed.add(i)
        except Exception:
            pass

    completed_count = 0
    while completed_count < len(metas) and completed_count in completed:
        completed_count += 1
    total_units = sum(int(m.get("max_iters") or 0) for m in metas)
    completed_units = sum(int(metas[i].get("max_iters") or 0) for i in range(completed_count))
    if completed_count >= len(metas):
        return {
            "source": "bapr_seed_batch",
            "eta_s": 0,
            "total_s": int(max(0, elapsed_s)),
            "current": total_units,
            "total_units": total_units,
            "unit_s": (float(elapsed_s) / float(total_units)) if total_units > 0 else None,
        }
    if current is None:
        return None
    current_total = int(metas[completed_count].get("max_iters") or 0)
    current = max(0, min(int(current), current_total))
    current_units = completed_units + current
    if total_units <= 0 or current_units <= 0 or elapsed_s <= 0:
        return None
    unit_s = float(elapsed_s) / float(current_units)
    eta_s = int(max(0, (total_units - current_units) * unit_s))
    return {
        "source": "bapr_seed_batch",
        "eta_s": eta_s,
        "total_s": int(max(elapsed_s, total_units * unit_s)),
        "current": int(current_units),
        "total_units": int(total_units),
        "unit_s": unit_s,
        "completed_seeds": completed_count,
        "seed_count": len(metas),
    }

def _infer_checkpoint_from_submit(cmd: str, cwd: str = ""):
    return (_infer_direct_jax_train_ckpt(cmd, cwd)
            or _infer_wrapper_ckpt(cmd, cwd)
            or _infer_bapr_run_seed_ckpt(cmd, cwd))

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
        int(task.get("cpu_parallel_total_items") or task.get("cpu_parallel_items") or 0),
        int(task.get("cpu_parallel_start") or 0),
        int(task.get("cpu_parallel_end") or 0),
        int(task.get("cpu_auto_workers") or 0),
    )


def _local_launch_transport_alive(task: dict) -> bool:
    """True when the local SSH transport that owns a Windows foreground launch still exists."""
    pid = task.get("local_ssh_pid")
    if not pid:
        return False
    try:
        return alive(int(pid))
    except Exception:
        return False


def _task_has_recorded_launch_artifacts(task: dict) -> bool:
    return bool(
        _task_pids(task)
        or task.get("slurm_job_id")
        or task.get("local_ssh_pid")
        or (task.get("log_path") and task.get("started_at"))
    )


def _recorded_launch_safety_state(task: dict) -> tuple[str, str]:
    """Fail-closed liveness check used before requeueing or duplicate dispatch.

    Returns (alive|dead|unknown, reason). Unknown intentionally blocks a duplicate
    launch; transient SSH/Windows probe failures must behave like Linux's proc
    probe failure path, not like a terminal task.
    """
    if not _task_has_recorded_launch_artifacts(task):
        return "dead", "no recorded launch artifacts"
    if _local_launch_transport_alive(task):
        return "alive", f"local launch transport pid {task.get('local_ssh_pid')} is alive"
    terminal = task.get("status") in ("done", "failed", "cancelled", "forgotten")
    if terminal and task.get("finished_at") and not (_task_pids(task) or task.get("slurm_job_id")):
        return "dead", "terminal task has no backend pid/job; log artifact alone is not live"
    has_backend_artifact = bool(
        _task_pids(task)
        or task.get("slurm_job_id")
        or (task.get("log_path") and task.get("started_at"))
    )
    if not has_backend_artifact:
        return "dead", "only recorded local launch transport is gone"
    node = task.get("node") or task.get("last_node")
    if not node:
        return "unknown", "recorded launch artifacts exist but node/last_node is missing"
    if not (_task_pids(task) or task.get("slurm_job_id")):
        return "unknown", "recorded launch artifacts exist but no backend pid/job id is available"
    probe_task = dict(task)
    probe_task["status"] = "running"
    probe_task["node"] = node
    try:
        res = _BACKEND.batch_probe({"tasks": [probe_task]}).get(task.get("id"))
    except Exception as e:
        return "unknown", f"backend probe raised {str(e)[:160]}"
    if not res or res.get("state") == "unknown":
        err = (res or {}).get("error") or "backend probe returned unknown"
        return "unknown", str(err)[:200]
    if res.get("state") == "alive":
        fallback = str(res.get("probe_fallback") or "")
        if terminal and (fallback == "windows_queue_accounting" or fallback.startswith("windows_parse_failed")):
            return (
                "dead",
                f"terminal Windows task only has unverified liveness fallback {fallback}; "
                "treating recorded pid as stale",
            )
        return "alive", "backend probe reports recorded pid/job alive"
    if res.get("state") == "dead":
        return "dead", "backend probe reports recorded pid/job dead"
    return "unknown", f"unexpected backend state {res.get('state')!r}"


def _same_run_identity_live_artifact_reason(task: dict, state: dict, run_key=None) -> Optional[str]:
    key = run_key if run_key is not None else _task_run_identity(task)
    if not key:
        return None
    target_id = task.get("id")
    tasks = state.get("tasks", [])
    by_id = {t.get("id"): t for t in tasks if t.get("id")}
    ancestor_ids = set()
    cur = task
    seen = set()
    while cur:
        pid = cur.get("parent_id")
        if not pid or pid in seen:
            break
        seen.add(pid)
        ancestor_ids.add(pid)
        cur = by_id.get(pid)
    for other in tasks:
        if other is task or other.get("id") == target_id:
            continue
        if other.get("id") in ancestor_ids:
            # A retry clone intentionally represents the same run identity as
            # its parent. Old launch handles retained on the terminal lineage
            # are audit data, not a reason to block that lineage's next retry.
            continue
        if _task_run_identity(other) != key:
            continue
        if other.get("status") in ("queued", "running", "launching"):
            continue
        if not _task_has_recorded_launch_artifacts(other):
            continue
        launch_state, launch_reason = _recorded_launch_safety_state(other)
        if launch_state in ("alive", "unknown"):
            return (
                f"run identity has {launch_state} launch artifacts in terminal task "
                f"{other.get('id')} ({launch_reason}); refusing duplicate dispatch"
            )
    return None


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
    actor = _actor_info("cancel", reason)
    task["status"] = "cancelled"
    task["finished_at"] = now
    task["cancelled_at"] = now
    task["cancelled_by_user"] = True
    task["cancelled_by"] = actor["label"]
    task["cancel_actor"] = actor
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
    if task.get("local_ssh_pid"):
        artifacts.append("local_ssh_pid")
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


def _runtime_device_kind_for_task(task: dict, gpu_idx=None) -> str:
    if gpu_idx is None or _task_launch_cpu_mode(task):
        return "cpu"
    return "gpu"


def _runtime_node_bucket_key(node: str, device_kind: str) -> str:
    return f"{node or '?'}:{device_kind or '?'}"


def _candidate_runtime_seconds(task: dict, node: str, gpu_idx=None) -> int:
    """Return node/device-specific runtime history for placement scoring, or 0 if unknown."""
    h = load_runtime_history()
    device = _runtime_device_kind_for_task(task, gpu_idx)
    bucket = _runtime_node_bucket_key(node, device)
    for key, _kind, _payload in _task_runtime_keys(task):
        rec = h.get(key)
        if not isinstance(rec, dict):
            continue
        node_runtime = rec.get("node_runtime") or {}
        b = node_runtime.get(bucket)
        if isinstance(b, dict) and int(b.get("total_s") or 0) > 0:
            return int(b.get("total_s") or 0)
    return 0


def _load_eta_tracker_module():
    try:
        from . import eta_tracker  # type: ignore
        return eta_tracker
    except Exception:
        try:
            import importlib.util as _ilu  # type: ignore
            spec = _ilu.spec_from_file_location(
                "eta_tracker", str(Path(__file__).parent / "eta_tracker.py")
            )
            if spec and spec.loader:
                mod = _ilu.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
        except Exception:
            return None
    return None


def _runtime_total_units_from_cmd(cmd: str) -> int:
    et = _load_eta_tracker_module()
    if et and hasattr(et, "_extract_total_from_cmd"):
        try:
            return int(et._extract_total_from_cmd(cmd) or 0)
        except Exception:
            return 0
    return 0


def _runtime_cmd_tokens(cmd: str) -> set:
    try:
        toks = shlex.split(cmd or "")
    except Exception:
        toks = (cmd or "").split()
    out = set()
    skip_next_for = {
        "--seed", "--current_time", "--current-time", "--run_name", "--run-name",
        "--output", "--out", "--log_dir", "--log-dir",
    }
    skip_next = False
    for tok in toks:
        if skip_next:
            skip_next = False
            continue
        if tok in skip_next_for:
            out.add(tok)
            skip_next = True
            continue
        if any(tok.startswith(f + "=") for f in skip_next_for):
            out.add(tok.split("=", 1)[0])
            continue
        base = os.path.basename(tok)
        if base in ("python", "python3", "python3.10", "python3.11", "python3.12", "bash", "sh"):
            continue
        norm = re.sub(r"\d+", "<N>", tok.lower())
        if len(norm) >= 2:
            out.add(norm)
    return out


def _runtime_script_name(cmd: str) -> str:
    try:
        toks = shlex.split(cmd or "")
    except Exception:
        toks = (cmd or "").split()
    for tok in toks:
        if tok.endswith((".py", ".sh")) or ".py:" in tok:
            return os.path.basename(tok)
    return ""


def _runtime_history_closest(task: dict, history: dict):
    """Return a conservative closest runtime-history record for novel signatures.

    This is the deterministic fallback for the "AI should choose closest task"
    rule: exact cmd/cwd and explicit signature still win. Closest only engages
    when project/cwd/script/tokens are sufficiently similar, and records its
    source so the estimate is auditable.
    """
    payload = _task_runtime_payload(task)
    task_tokens = _runtime_cmd_tokens(payload.get("cmd") or "")
    if not task_tokens:
        return None, None
    task_script = _runtime_script_name(payload.get("cmd") or "")
    task_project = payload.get("project") or ""
    task_cwd_base = os.path.basename(payload.get("cwd") or "")
    task_units = _runtime_total_units_from_cmd(payload.get("cmd") or "")
    best = None
    for key, rec in (history or {}).items():
        if not isinstance(rec, dict) or int(rec.get("total_s") or 0) <= 0:
            continue
        rec_cmd = rec.get("cmd") or ""
        rec_tokens = _runtime_cmd_tokens(rec_cmd)
        if not rec_tokens:
            continue
        inter = len(task_tokens & rec_tokens)
        union = len(task_tokens | rec_tokens) or 1
        score = inter / union
        rec_script = _runtime_script_name(rec_cmd)
        if task_script and rec_script and task_script == rec_script:
            score += 0.25
        if task_project and rec.get("project") == task_project:
            score += 0.15
        rec_cwd_base = os.path.basename(str(rec.get("cwd") or ""))
        if task_cwd_base and rec_cwd_base and task_cwd_base == rec_cwd_base:
            score += 0.10
        if score < RUNTIME_CLOSEST_MIN_SCORE:
            continue
        if best is None or score > best[0]:
            best = (score, key, rec)
    if not best:
        return None, None
    score, key, rec = best
    out = dict(rec)
    out["source"] = f"closest:{key}:score={score:.2f}"
    if task_units > 0 and float(rec.get("unit_s") or 0) > 0:
        out["total_s"] = int(float(rec["unit_s"]) * float(task_units))
        out["walltime_s"] = int(max(RUNTIME_MIN_WALLTIME_S, out["total_s"] * RUNTIME_WALLTIME_MULT))
        out["total_units"] = task_units
    return out, f"closest:{key}"


def _runtime_history_best(task: dict):
    h = load_runtime_history()
    for key, kind, _payload in _task_runtime_keys(task):
        rec = h.get(key)
        if isinstance(rec, dict) and int(rec.get("total_s") or 0) > 0:
            return rec, key, kind
    rec, key = _runtime_history_closest(task, h)
    if rec:
        return rec, key, "closest"
    return None, None, None


def _runtime_total_history_s(task: dict) -> int:
    rec, _key, _kind = _runtime_history_best(task)
    return int(rec.get("total_s") or 0) if rec else 0


def _history_eta_for_task(task: dict) -> tuple[int, str]:
    """Return a full-run ETA estimate for a task that has not started yet.

    Running tasks get live log-tail ETA from _refresh_eta_from_logs. Queued and
    launching tasks have no elapsed progress, so their ETA is the best known
    total runtime: local preflight profile on the task, exact/closest runtime
    history, then legacy duration EWMA.
    """
    total_s = int(task.get("runtime_total_s_est") or 0)
    if total_s > 0:
        return total_s, task.get("runtime_est_source") or "runtime_profile"
    total_s = _runtime_total_history_s(task)
    if total_s > 0:
        return total_s, "runtime_history"
    sig = task.get("signature") or ""
    h = history_get(sig) or {}
    total_s = int(h.get("dur_s_ewma") or 0)
    if total_s > 0:
        return total_s, "duration_ewma"
    return 0, ""


def _seed_pending_eta_from_history(state: dict) -> int:
    """Fill ETA for queued/launching tasks from local-test/runtime history.

    This makes status/eta_load/migration see the same local preflight signal
    that Slurm walltime already uses. It only fills unknown ETA; live running
    ETA remains owned by log-tail progress parsing. If a queued task still
    carries live remaining-time from a previous process (preempt/requeue/
    rebalance), clear it first; queued ETA must be a full-run profile, not
    leftover remaining time from a killed process.
    """
    changed = 0
    for t in state.get("tasks", []):
        if t.get("status") not in ("queued", "launching"):
            continue
        if _queued_has_stale_live_eta(t):
            if _clear_live_eta_fields(t, clear_runtime_projection=True):
                t["eta_detail"] = "cleared stale live ETA after task returned to queue"
                changed += 1
        if int(t.get("eta_seconds") or 0) > 0:
            continue
        eta, source = _history_eta_for_task(t)
        if eta <= 0:
            continue
        t["eta_seconds"] = int(eta)
        t["eta_source"] = source
        t["eta_confidence"] = _eta_confidence_for_source(source)
        t["eta_updated_at"] = int(time.time())
        t["eta_detail"] = f"queued ETA seeded from {source}"
        changed += 1
    return changed


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
        node = task.get("node") or task.get("last_node") or ""
        if node:
            device = _runtime_device_kind_for_task(task, task.get("gpu_idx"))
            bucket = _runtime_node_bucket_key(node, device)
            node_runtime = cur.get("node_runtime")
            if not isinstance(node_runtime, dict):
                node_runtime = {}
            node_rec = node_runtime.get(bucket)
            if not isinstance(node_rec, dict):
                node_rec = {}
            ns = node_rec.get("total_s_samples") or []
            ns.append(int(total_s))
            ns = ns[-RUNTIME_HISTORY_SAMPLES_PER_KEY:]
            node_rec["total_s_samples"] = ns
            node_rec["total_s"] = _percentile(ns, RUNTIME_HISTORY_PERCENTILE)
            if unit_s > 0:
                nus = node_rec.get("unit_s_samples") or []
                nus.append(float(unit_s))
                nus = nus[-RUNTIME_HISTORY_SAMPLES_PER_KEY:]
                node_rec["unit_s_samples"] = nus
                node_rec["unit_s"] = float(_percentile([int(x * 1000) for x in nus], RUNTIME_HISTORY_PERCENTILE)) / 1000.0
            node_rec["runs"] = int(node_rec.get("runs") or 0) + 1
            node_rec["last_seen"] = now
            node_rec["device_kind"] = device
            node_runtime[bucket] = node_rec
            cur["node_runtime"] = node_runtime
        cur["source"] = source or cur.get("source") or "duration"
        cur["kind"] = kind
        cur["signature"] = payload.get("signature") or ""
        cur["project"] = payload.get("project") or ""
        cur["description"] = payload.get("description") or ""
        cur["cwd"] = payload.get("cwd") or ""
        cur["env_spec"] = payload.get("env_spec") or "none"
        cur["cmd"] = (payload.get("cmd") or "")[:500]
        cur["runs"] = int(cur.get("runs") or 0) + 1
        cur["last_seen"] = now
        h[key] = cur
    if len(h) > RUNTIME_HISTORY_MAX_ENTRIES:
        kept = sorted(h.items(), key=lambda kv: -(kv[1].get("last_seen", 0) if isinstance(kv[1], dict) else 0))
        h = dict(kept[:RUNTIME_HISTORY_MAX_ENTRIES])
    save_runtime_history(h)


def _read_text_tail(path: str, max_bytes: int = 2 * 1024 * 1024) -> str:
    try:
        with open(os.path.expanduser(path), "rb") as f:
            try:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - max_bytes), os.SEEK_SET)
            except OSError:
                pass
            return f.read(max_bytes).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _runtime_profile_from_log(log_path: str, cmd: str = "", observed_duration_s: float = 0):
    et = _load_eta_tracker_module()
    if not et or not hasattr(et, "runtime_projection_from_log"):
        return None
    text = _read_text_tail(log_path)
    if not text:
        return None
    try:
        return et.runtime_projection_from_log(
            text, cmd=cmd, observed_duration_s=observed_duration_s)
    except Exception:
        return None


def _apply_test_log_runtime_profile(task: dict, test_log: str) -> bool:
    profile = _runtime_profile_from_log(test_log, task.get("cmd") or "")
    if not profile:
        return False
    profile = dict(profile)
    profile["source"] = (profile.get("source") or "local_test_log") + ":" + os.path.expanduser(test_log)
    _apply_runtime_projection(task, profile)
    task["test_log_path"] = os.path.expanduser(test_log)
    runtime_history_record(task)
    return True


def _queued_cpu_training_block_reason(task):
    if task.get("status") != "queued":
        return None
    if task.get("auto_adopted") or task.get("adopted"):
        return None
    # A task hard-pinned to a configured CPU labor node is intentionally
    # CPU-only; do not require the separate guard flag during dispatch.
    req_node = task.get("require_node")
    if (req_node in _cpu_labor_node_names()
            and int(task.get("est_vram_mb") or 0) <= 0):
        return None
    return _cpu_training_policy_reason(
        _submit_policy_text(task.get("cmd", ""), task.get("cwd", "")),
        task.get("description", ""),
        bool(task.get("allow_cpu_training", False)),
        int(task.get("est_vram_mb") or 0),
    )

def _queued_wait_for_file_block_reason(task):
    if task.get("status") != "queued":
        return None
    waits = task.get("wait_for_files") or []
    if isinstance(waits, str):
        waits = [waits]
    if not waits:
        return None
    missing = []
    for raw in waits:
        path_s = os.path.expandvars(os.path.expanduser(str(raw)))
        try:
            st = os.stat(path_s)
        except OSError:
            missing.append(path_s)
            continue
        if not os.path.isfile(path_s) or st.st_size <= 0:
            missing.append(path_s)
    if not missing:
        return None
    shown = ", ".join(missing[:3])
    more = "" if len(missing) <= 3 else f" (+{len(missing) - 3} more)"
    return f"waiting for prerequisite file(s): {shown}{more}"

def _flag_values_any(tokens: list, flags: set) -> list:
    vals = []
    parsed = _cmd_flag_values(tokens, flags)
    for flag in flags:
        vals.extend(parsed.get(flag, []))
    return vals

def _flag_present_or_true(tokens: list, flag: str) -> bool:
    for tok in tokens:
        if tok == flag:
            return True
        if tok.startswith(flag + "="):
            val = tok.split("=", 1)[1].strip().lower()
            return val not in ("0", "false", "no", "off")
    return False

def _simple_sac_large_data_reason(cmd: str, cwd: str) -> Optional[str]:
    """Return a reason when SimpleSAC should stay local due to external data.

    scheduleurm's launch staging rsyncs the task cwd only. SimpleSAC bus training
    reads H2Oplus/bus_h2o outside the cwd; snapshot/per-file modes can involve
    ~122GB of HDF5 archives, so they must not be silently routed to a server.
    """
    text = cmd or ""
    if "h2o+_bus_main.py" not in text:
        return None
    try:
        tokens = shlex.split(text)
    except Exception:
        tokens = text.split()

    dataset_dir_vals = _flag_values_any(tokens, {"--dataset_dir", "--dataset-dir"})
    if any(str(v).strip() for v in dataset_dir_vals):
        return ("SimpleSAC bus training uses --dataset_dir/per-policy HDF5 data "
                "outside cwd; local bus_h2o/datasets_v2 is about 122GB and is not "
                "part of scheduler cwd staging")

    snapshot_disabled = any(tok == "--nouse_snapshot_reset" for tok in tokens)
    for flag in ("--use_snapshot_reset", "--use-snapshot-reset"):
        for tok in tokens:
            if tok.startswith(flag + "="):
                val = tok.split("=", 1)[1].strip().lower()
                if val in ("0", "false", "no", "off"):
                    snapshot_disabled = True

    if not snapshot_disabled:
        return ("SimpleSAC snapshot reset is enabled by default; snapshot-capable "
                "runs may require the external bus_h2o/datasets_v2 archive "
                "(about 122GB), which scheduler does not rsync with cwd")

    return None

def _doctor_path_key(path: str) -> str:
    return os.path.normpath(os.path.expandvars(os.path.expanduser(str(path or ""))))

def _resolve_cmd_path(raw: str, cwd: str = "") -> str:
    raw = os.path.expandvars(os.path.expanduser(str(raw or "")))
    if not raw:
        return ""
    if os.path.isabs(raw):
        return os.path.normpath(raw)
    return os.path.normpath(os.path.join(cwd or os.getcwd(), raw))

def _task_wait_files(task: dict) -> list:
    waits = task.get("wait_for_files") or []
    if isinstance(waits, str):
        waits = [waits]
    return [str(w) for w in waits if str(w or "").strip()]

def _task_has_wait_file(task: dict, path: str) -> bool:
    want = _doctor_path_key(path)
    return any(_doctor_path_key(w) == want for w in _task_wait_files(task))

def _ready_local_file(path: str) -> bool:
    try:
        st = os.stat(_doctor_path_key(path))
        return os.path.isfile(_doctor_path_key(path)) and st.st_size > 0
    except OSError:
        return False

def _simple_sac_ckpt_for_method(script_path: str, method: str) -> Optional[str]:
    """Infer run_multiseed_eval.sh's checkpoint path for a method tag.

    This intentionally parses only the local, simple case-table format used by
    SimpleSAC's eval wrapper. If the wrapper changes enough that we cannot infer
    the path, doctor reports nothing instead of inventing a prerequisite.
    """
    script_path = _resolve_cmd_path(script_path)
    if not method or not os.path.isfile(script_path):
        return None
    src = _safe_read_text(script_path)
    if not src:
        return None
    pat = r"(?ms)^\s*" + re.escape(method) + r"\)\s*(.*?)^\s*;;"
    m = re.search(pat, src)
    if not m:
        return None
    block = m.group(1)
    cm = re.search(r"CKPT=(?:\"([^\"]*)\"|'([^']*)'|([^\s#;]+))", block)
    if not cm:
        return None
    raw = next((g for g in cm.groups() if g is not None), "")
    raw = raw.strip()
    if not raw:
        return None
    here = os.path.dirname(script_path)
    h2o_root = os.path.dirname(here)
    repl = {
        "$H2O_ROOT": h2o_root,
        "${H2O_ROOT}": h2o_root,
        "$HERE": here,
        "${HERE}": here,
        "$HOME": str(Path.home()),
        "${HOME}": str(Path.home()),
    }
    for key, val in repl.items():
        raw = raw.replace(key, val)
    return _resolve_cmd_path(raw, here)

def _simple_sac_eval_prereq_files(cmd: str, cwd: str) -> list:
    """Return checkpoint files an eval command needs before dispatch.

    Covers the two patterns that caused wasted work:
      * SimpleSAC run_multiseed_eval.sh METHOD ...
      * direct eval scripts with --checkpoint PATH
    """
    try:
        toks = shlex.split(cmd or "")
    except Exception:
        toks = (cmd or "").split()
    if not toks:
        return []

    prereqs = []
    for i, tok in enumerate(toks):
        if os.path.basename(tok) != "run_multiseed_eval.sh":
            continue
        if i + 1 >= len(toks):
            continue
        script = _resolve_cmd_path(tok, cwd)
        ckpt = _simple_sac_ckpt_for_method(script, toks[i + 1])
        if ckpt:
            prereqs.append(ckpt)

    lower = " ".join(toks).lower()
    if "eval" in lower:
        ckpt_vals = _flag_values_any(
            toks,
            {"--checkpoint", "--ckpt", "--ckpt_path", "--ckpt-path"},
        )
        for v in ckpt_vals:
            p = _resolve_cmd_path(v, cwd)
            if p:
                prereqs.append(p)

    out = []
    seen = set()
    for p in prereqs:
        k = _doctor_path_key(p)
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out

def _simple_sac_train_best_ckpt_from_cmd(cmd: str, cwd: str) -> Optional[str]:
    """Infer h2o+_bus_main.py's final best checkpoint path from seed/current_time."""
    if "h2o+_bus_main.py" not in (cmd or ""):
        return None
    try:
        toks = shlex.split(cmd or "")
    except Exception:
        toks = (cmd or "").split()
    seed = _arg_value(toks, "--seed")
    current_time = _arg_value(toks, "--current_time") or _arg_value(toks, "--current-time")
    if not seed or not current_time:
        return None
    h2o_root = os.path.dirname(os.path.normpath(cwd or os.getcwd()))
    return os.path.join(
        h2o_root,
        "experiment_output",
        f"h2oplus_bus_seed{seed}_{current_time}",
        "checkpoint_best.pt",
    )

def _doctor_issue(task: Optional[dict], code: str, severity: str,
                  message: str, fix: str = "", fixed: bool = False,
                  path: str = "") -> dict:
    rec = {
        "code": code,
        "severity": severity,
        "message": message,
        "fix": fix,
        "fixed": bool(fixed),
    }
    if task:
        rec["task_id"] = task.get("id")
        rec["status"] = task.get("status")
        rec["project"] = task.get("project")
        rec["signature"] = task.get("signature")
    if path:
        rec["path"] = path
    return rec

def _doctor_scan_state(state: dict, fix: bool = False, project: str = ""):
    """Audit active queue invariants and optionally repair safe queued-task fields.

    Safe fixes are intentionally narrow:
      * add wait_for_files to queued eval tasks whose checkpoint dependency is known
      * force queued SimpleSAC large-data training local
      * promote queued SimpleSAC training to high when queued evals depend on its ckpt

    Running/launching tasks are never mutated here.
    """
    import fnmatch as _fnmatch

    def selected(t):
        if not project:
            return True
        return _fnmatch.fnmatch(t.get("project") or "", project)

    tasks = [t for t in state.get("tasks", []) if selected(t)]
    active = [t for t in tasks if t.get("status") in ("queued", "launching", "running")]
    issues = []
    changed = 0

    needed_paths = {}
    producer_by_path = {}
    for t in active:
        train_ckpt = _simple_sac_train_best_ckpt_from_cmd(t.get("cmd") or "", t.get("cwd") or "")
        if train_ckpt:
            producer_by_path[_doctor_path_key(train_ckpt)] = t

    for t in active:
        prereqs = _simple_sac_eval_prereq_files(t.get("cmd") or "", t.get("cwd") or "")
        for pth in prereqs:
            needed_paths.setdefault(_doctor_path_key(pth), []).append(t)
        missing_waits = [p for p in prereqs if not _task_has_wait_file(t, p)]
        if not missing_waits:
            continue
        if t.get("status") == "queued":
            if fix:
                waits = _task_wait_files(t)
                waits.extend(p for p in missing_waits if not _task_has_wait_file({"wait_for_files": waits}, p))
                t["wait_for_files"] = waits
                t["last_block_reason"] = _queued_wait_for_file_block_reason(t) or t.get("last_block_reason")
                changed += 1
                issues.append(_doctor_issue(
                    t, "eval_missing_wait_for_file", "fixed",
                    "queued eval was missing checkpoint prerequisite gating",
                    fixed=True, path=", ".join(missing_waits),
                ))
            else:
                issues.append(_doctor_issue(
                    t, "eval_missing_wait_for_file", "fixable",
                    "queued eval can dispatch before its checkpoint exists",
                    "doctor --fix will add wait_for_files", path=", ".join(missing_waits),
                ))
        else:
            issues.append(_doctor_issue(
                t, "eval_already_running_without_wait_for_file", "warn",
                "eval is already launching/running without scheduler-level checkpoint gating",
                "inspect log; cancel/requeue manually if it is burning time on a missing ckpt",
                path=", ".join(missing_waits),
            ))

    for t in active:
        reason = _simple_sac_large_data_reason(t.get("cmd") or "", t.get("cwd") or "")
        if not reason or t.get("allow_remote_large_data"):
            continue
        if t.get("status") == "queued":
            if t.get("require_node") == "local":
                continue
            if fix:
                old_node = t.get("node")
                t["require_node"] = "local"
                if t.get("preferred_node") != "local":
                    t["preferred_node"] = None
                t["node"] = None
                t["gpu_idx"] = None
                t["remote_pids"] = []
                t["last_block_reason"] = f"doctor: forced local for SimpleSAC large data: {reason}"
                try:
                    _release_task_claims_and_intents(t, extra_nodes=[old_node] if old_node else None)
                except Exception:
                    pass
                changed += 1
                issues.append(_doctor_issue(
                    t, "simple_sac_large_data_not_local", "fixed",
                    reason, fixed=True,
                ))
            else:
                issues.append(_doctor_issue(
                    t, "simple_sac_large_data_not_local", "fixable",
                    reason,
                    "doctor --fix will set require_node=local",
                ))
        elif t.get("node") != "local":
            issues.append(_doctor_issue(
                t, "simple_sac_large_data_running_remote", "error",
                reason,
                "running task cannot be safely rewritten; cancel manually if this is wrong",
            ))

    for pth, eval_tasks in sorted(needed_paths.items()):
        producer = producer_by_path.get(pth)
        if producer and producer.get("status") == "queued" and producer.get("priority") != "high":
            if fix:
                producer["priority"] = "high"
                producer["last_block_reason"] = (
                    f"doctor: promoted training because eval task(s) wait for {pth}"
                )
                changed += 1
                issues.append(_doctor_issue(
                    producer, "train_priority_below_dependent_eval", "fixed",
                    "training produces a checkpoint needed by queued evals but was not high priority",
                    fixed=True, path=pth,
                ))
            else:
                issues.append(_doctor_issue(
                    producer, "train_priority_below_dependent_eval", "fixable",
                    "training produces a checkpoint needed by queued evals but is not high priority",
                    "doctor --fix will set priority=high", path=pth,
                ))
        if not _ready_local_file(pth) and not producer:
            ids = ", ".join(t.get("id", "?") for t in eval_tasks[:4])
            more = "" if len(eval_tasks) <= 4 else f" (+{len(eval_tasks) - 4} more)"
            issues.append(_doctor_issue(
                None, "eval_checkpoint_missing_no_active_producer", "warn",
                f"checkpoint is missing and no active SimpleSAC producer was inferred; eval tasks: {ids}{more}",
                "submit/restore the matching train task, or remove the evals if obsolete",
                path=pth,
            ))

    for t in active:
        if not _queued_has_stale_live_eta(t):
            continue
        msg = (
            f"queued task carries stale live ETA/progress "
            f"(eta_source={t.get('eta_source')!r}, runtime_est_source={t.get('runtime_est_source')!r}); "
            "queued ETA must be reseeded from local-test/runtime history"
        )
        if t.get("status") == "queued" and fix:
            _clear_live_eta_fields(t, clear_runtime_projection=True)
            _seed_pending_eta_from_history({"tasks": [t]})
            changed += 1
            issues.append(_doctor_issue(
                t, "queued_stale_live_eta", "fixed",
                msg, fixed=True,
            ))
        else:
            issues.append(_doctor_issue(
                t, "queued_stale_live_eta", "fixable",
                msg,
                "doctor --fix will clear live ETA fields and reseed from history",
            ))

    return issues, changed

def cmd_doctor(args):
    with state_lock():
        state = load_state()
        issues, changed = _doctor_scan_state(
            state,
            fix=bool(getattr(args, "fix", False)),
            project=getattr(args, "project", "") or "",
        )
        if changed:
            save_state(state)
    result = {"ok": True, "fixed": changed, "issues": issues}
    if args.json:
        print(json.dumps(result, indent=2))
        return
    if not issues:
        print("doctor: no active queue invariant issues found")
        return
    print(f"doctor: {len(issues)} issue(s), {changed} fixed")
    for issue in issues:
        tid = issue.get("task_id") or "-"
        path = f" path={issue['path']}" if issue.get("path") else ""
        print(f"  [{issue['severity']}] {issue['code']} {tid}: {issue['message']}{path}")
        if issue.get("fix") and not issue.get("fixed"):
            print(f"      fix: {issue['fix']}")


def cmd_profile_local(args):
    """Run a local preflight command outside the scheduler queue and record profile history."""
    cwd = os.path.abspath(os.path.expanduser(args.cwd))
    if not os.path.isdir(cwd):
        sys.exit(f"cwd does not exist: {cwd}")
    project = args.project or _project_from_path(cwd) or (
        args.signature.split("/", 1)[0] if "/" in args.signature else args.signature)
    log_path = os.path.expanduser(args.log_path or "")
    if not log_path:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.signature).strip("_") or "profile"
        log_path = str(LOG_DIR / f"profile_{safe}_{int(time.time())}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    extra_env = _parse_env(args.env)
    env = os.environ.copy()
    env.update(extra_env)
    inner = _inject_python_u(args.cmd)
    shell_cmd = f"cd {shlex.quote(cwd)} && {inner}"

    peak_vram = 0
    peak_ram = 0
    peak_cpu = 1
    start = time.time()
    with open(log_path, "ab", buffering=0) as log_f:
        header = (
            f"\n=== scheduleurm profile-local start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
            f"signature: {args.signature}\n"
            f"cwd: {cwd}\n"
            f"cmd: {args.cmd}\n"
        )
        log_f.write(header.encode("utf-8", errors="replace"))
        proc = subprocess.Popen(
            ["bash", "-lc", shell_cmd],
            cwd=cwd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid,
        )
        task = {
            "id": "profile-local",
            "status": "running",
            "node": "local",
            "remote_pids": [proc.pid],
            "process_group": proc.pid,
            "cmd": args.cmd,
            "cwd": cwd,
            "signature": args.signature,
            "project": project,
            "description": args.description or "local preflight profile",
            "extra_env": extra_env,
            "env_spec": "none",
            "image": "",
            "started_at": start,
        }
        backend = LocalBackend()
        interval = max(1.0, float(args.sample_interval))
        try:
            while True:
                res = backend.batch_probe({"tasks": [task]}).get("profile-local")
                if res and res.get("state") == "alive":
                    peak_vram = max(peak_vram, int(res.get("vram_mb") or 0))
                    peak_ram = max(peak_ram, int(res.get("ram_mb") or 0))
                    pcpu = float(res.get("pcpu") or 0.0)
                    peak_cpu = max(peak_cpu, max(1, int((pcpu + 99.0) // 100.0)))
                rc = proc.poll()
                if rc is not None:
                    break
                if args.timeout and time.time() - start > args.timeout:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                        time.sleep(2)
                        if proc.poll() is None:
                            os.killpg(proc.pid, signal.SIGKILL)
                    except Exception:
                        pass
                    rc = proc.wait()
                    break
                time.sleep(interval)
        finally:
            rc = proc.poll()
            if rc is None:
                rc = proc.wait()
        footer = (
            f"\n=== scheduleurm profile-local end rc={rc} "
            f"duration_s={int(time.time() - start)} peak_vram_mb={peak_vram} "
            f"peak_ram_mb={peak_ram} peak_cpu={peak_cpu} ===\n"
        )
        log_f.write(footer.encode("utf-8", errors="replace"))

    duration_s = int(max(0, time.time() - start))
    profile_task = {
        "id": "profile-local",
        "signature": args.signature,
        "project": project,
        "description": args.description or "local preflight profile",
        "cmd": args.cmd,
        "cwd": cwd,
        "env_spec": "none",
        "image": "",
        "extra_env": extra_env,
    }
    profile = _runtime_profile_from_log(log_path, args.cmd, observed_duration_s=duration_s)
    if profile:
        profile = dict(profile)
        profile["source"] = (profile.get("source") or "profile-local") + ":" + log_path
        _apply_runtime_projection(profile_task, profile)
    history_record(
        args.signature,
        peak_vram_mb=peak_vram,
        peak_ram_mb=peak_ram,
        cpu_cores=peak_cpu,
        duration_s=duration_s,
    )
    runtime_history_record(profile_task, duration_s=duration_s)

    out = {
        "ok": rc == 0,
        "exit_code": rc,
        "log_path": log_path,
        "duration_s": duration_s,
        "peak_vram_mb": peak_vram,
        "peak_ram_mb": peak_ram,
        "peak_cpu_cores": peak_cpu,
        "runtime_profile": {
            "source": profile_task.get("runtime_est_source") or ("duration" if duration_s else ""),
            "total_s": profile_task.get("runtime_total_s_est") or duration_s,
            "eta_s": profile_task.get("runtime_eta_s_est"),
            "unit_s": profile_task.get("runtime_unit_s_est"),
            "total_units": profile_task.get("runtime_total_units"),
        },
    }
    if args.json:
        print(json.dumps(out, indent=2))
        return
    print(f"profile-local: rc={rc} log={log_path}")
    print(f"  duration={duration_s}s peak_vram={peak_vram}MB peak_ram={peak_ram}MB peak_cpu={peak_cpu}")
    rp = out["runtime_profile"]
    if rp.get("total_s"):
        print(f"  runtime_total={rp.get('total_s')}s source={rp.get('source')}")

def cmd_submit(args):
    split_args = _split_bapr_seed_batch_submit_args(args)
    if split_args:
        print(f"NOTE: auto-splitting BAPR seed batch into {len(split_args)} per-seed tasks",
              file=sys.stderr)
        for child_args in split_args:
            cmd_submit(child_args)
        return

    raw_submit_cmd = args.cmd
    policy_cmd = _submit_policy_text(args.cmd, args.cwd)
    inferred_checkpoint = _infer_checkpoint_from_submit(args.cmd, args.cwd)
    inferred_ckpt_dir = ""
    inferred_result_dir = ""
    inferred_resume_managed = False
    inferred_ckpt_source = ""
    if inferred_checkpoint:
        inferred_ckpt_dir = inferred_checkpoint.get("ckpt_dir") or ""
        inferred_result_dir = inferred_checkpoint.get("result_dir") or ""
        inferred_resume_managed = bool(inferred_checkpoint.get("resume_managed_by_cmd"))
        inferred_ckpt_source = inferred_checkpoint.get("source") or ""
    ckpt_dir_was_inferred = False
    if not args.ckpt_dir and inferred_ckpt_dir:
        args.ckpt_dir = inferred_ckpt_dir
        ckpt_dir_was_inferred = True
    if not getattr(args, "result_dir", None) and inferred_result_dir:
        args.result_dir = inferred_result_dir

    large_data_reason = _simple_sac_large_data_reason(raw_submit_cmd, args.cwd)
    if large_data_reason and not getattr(args, "allow_remote_large_data", False):
        if args.require_node and args.require_node != "local":
            print("REFUSED: task appears to require large local-only SimpleSAC data.",
                  file=sys.stderr)
            print(f"  reason: {large_data_reason}", file=sys.stderr)
            print("  Either keep --require-node local, disable snapshot/per-file data, "
                  "or pass --allow-remote-large-data after manually staging the data.",
                  file=sys.stderr)
            sys.exit(2)
        if args.preferred_node and args.preferred_node != "local":
            print("REFUSED: task prefers a remote node but appears to require large "
                  "local-only SimpleSAC data.", file=sys.stderr)
            print(f"  reason: {large_data_reason}", file=sys.stderr)
            print("  Either keep it local, disable snapshot/per-file data, or pass "
                  "--allow-remote-large-data after manually staging the data.",
                  file=sys.stderr)
            sys.exit(2)
        if args.require_node != "local":
            args.require_node = "local"
            print(f"NOTE: forcing --require-node local: {large_data_reason}",
                  file=sys.stderr)

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
    test_peak_vram = int(getattr(args, "test_peak_vram_mb", 0) or 0)
    test_peak_ram = int(getattr(args, "test_peak_ram_mb", 0) or 0)
    test_cpu = int(getattr(args, "test_cpu", 0) or 0)
    cpu_parallel_items = int(getattr(args, "cpu_parallel_items", 0) or 0)
    cpu_parallel_total_items = int(getattr(args, "cpu_parallel_total_items", 0) or 0) or cpu_parallel_items
    cpu_parallel_logical_items = int(getattr(args, "cpu_parallel_logical_items", 0) or 0)
    if cpu_parallel_items and not cpu_parallel_logical_items:
        cpu_parallel_logical_items = cpu_parallel_total_items
    cpu_parallel_item_multiplier = int(getattr(args, "cpu_parallel_item_multiplier", 1) or 1)
    cpu_parallel_start = int(getattr(args, "cpu_parallel_start", 0) or 0)
    cpu_parallel_end = int(getattr(args, "cpu_parallel_end", 0) or 0) or cpu_parallel_items
    cpu_parallel_shard_index = int(getattr(args, "cpu_parallel_shard_index", 0) or 0)
    cpu_parallel_num_shards = int(getattr(args, "cpu_parallel_num_shards", 1) or 1)
    cpu_batch_plan = getattr(args, "cpu_batch_plan", None)
    allowed_nodes = list(getattr(args, "allowed_nodes", None) or [])
    stage_excludes = [str(x).strip() for x in (getattr(args, "stage_excludes", None) or [])
                      if str(x).strip()]
    if test_peak_vram > 0 or test_peak_ram > 0 or test_cpu > 0:
        history_record(
            args.signature,
            peak_vram_mb=test_peak_vram,
            peak_ram_mb=test_peak_ram,
            cpu_cores=test_cpu,
        )
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
                "cpu_parallel_items": cpu_parallel_items,
                "cpu_parallel_total_items": cpu_parallel_total_items,
                "cpu_parallel_logical_items": cpu_parallel_logical_items,
                "cpu_parallel_item_multiplier": cpu_parallel_item_multiplier,
                "cpu_parallel_start": cpu_parallel_start,
                "cpu_parallel_end": cpu_parallel_end,
                "allowed_nodes": allowed_nodes,
            })
            for existing in state["tasks"]:
                if _task_run_identity(existing) != submit_identity:
                    continue
                if existing.get("status") in ("queued", "running", "launching"):
                    print(f"DUPLICATE: {existing['id']} ({existing['status']}) has identical run identity")
                    print(f"  signature: {sig}")
                    print(f"  cmd: {args.cmd[:120]}")
                    print(f"  cwd: {args.cwd}")
                    print(f"  pass --allow-duplicate to override")
                    sys.exit(2)
                if _task_has_recorded_launch_artifacts(existing):
                    launch_state, launch_reason = _recorded_launch_safety_state(existing)
                    if launch_state in ("alive", "unknown"):
                        print(
                            f"DUPLICATE: {existing['id']} ({existing.get('status')}) "
                            f"has {launch_state} launch artifacts for identical run identity"
                        )
                        print(f"  reason: {launch_reason}")
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
        cpu_auto_plan = None
        if args.cpu is not None:
            cpu_cores = args.cpu
            if cpu_parallel_items > 0:
                if isinstance(cpu_batch_plan, dict):
                    cpu_auto_plan = {
                        "workers": int(cpu_batch_plan.get("workers") or cpu_cores),
                        "waves": int(cpu_batch_plan.get("waves") or _ceil_div(cpu_parallel_items, max(1, cpu_cores))),
                        "physical_cores": int(cpu_batch_plan.get("physical_cores") or cpu_cores),
                        "total_physical_cores": int(cpu_batch_plan.get("total_physical_cores")
                                                    or cpu_batch_plan.get("physical_cores") or cpu_cores),
                        "last_wave_items": int(cpu_batch_plan.get("last_wave_items") or cpu_parallel_items),
                    }
                else:
                    cpu_auto_plan = _cpu_worker_plan_for_items(cpu_parallel_items, max(1, cpu_cores))
        elif cpu_parallel_items > 0:
            ref_node = args.require_node or args.preferred_node
            if ref_node:
                physical = _node_physical_cores(ref_node)
            else:
                physical = max([_node_physical_cores(n) for n in _cpu_labor_node_names()] or [DEFAULT_CPU_CORES])
            cpu_auto_plan = _cpu_worker_plan_for_items(cpu_parallel_items, physical)
            cpu_cores = cpu_auto_plan["workers"]
        else:
            cpu_cores = hist.get("cpu_cores", DEFAULT_CPU_CORES)
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
            "est_vram_mb_explicit": args.vram is not None,
            "ram_mb_explicit": args.ram_mb is not None,
            "cpu_cores_explicit": args.cpu is not None,
            "priority": args.priority,
            "preferred_node": args.preferred_node,
            "require_node": args.require_node,
            "allowed_nodes": allowed_nodes or None,
            "stage_excludes": stage_excludes or None,
            "reroute_on_node_down": bool(getattr(args, "reroute_on_node_down", False)),
            "node_down_requeue_s": int(getattr(args, "node_down_requeue_s", 0) or 0) or None,
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
            "wait_for_files": list(getattr(args, "wait_for_files", None) or []),
            "result_synced_at": None,
            "result_sync_error": None,
            "result_sync_attempts": 0,
            # Phase 3.4.11 P2: claim marker for concurrent-rsync prevention.
            # Set by _sync_completed_results_outside_lock under state_lock
            # before rsync; cleared on commit phase. Stale claims older
            # than RESULT_SYNC_TIMEOUT_S + grace are reclaimable.
            "result_syncing_at": None,
            "extra_env": extra_env,
            "origin": "scheduleurm",
            "submitted_by": _local_user(),
            "submitted_host": _local_host_short(),
            "scheduler_id": _ClaimManager.scheduler_id(),
            "cpu_parallel_items": cpu_parallel_items or None,
            "cpu_parallel_total_items": cpu_parallel_total_items if cpu_parallel_items else None,
            "cpu_parallel_logical_items": cpu_parallel_logical_items if cpu_parallel_items else None,
            "cpu_parallel_item_multiplier": cpu_parallel_item_multiplier if cpu_parallel_items else None,
            "cpu_parallel_start": cpu_parallel_start if cpu_parallel_items else None,
            "cpu_parallel_end": cpu_parallel_end if cpu_parallel_items else None,
            "cpu_parallel_shard_index": cpu_parallel_shard_index if cpu_parallel_items else None,
            "cpu_parallel_num_shards": cpu_parallel_num_shards if cpu_parallel_items else None,
            "cpu_auto_workers": (cpu_auto_plan or {}).get("workers") if cpu_auto_plan else None,
            "cpu_parallel_waves": (cpu_auto_plan or {}).get("waves") if cpu_auto_plan else None,
            "cpu_parallel_physical_cores": (cpu_auto_plan or {}).get("physical_cores") if cpu_auto_plan else None,
            "cpu_parallel_total_physical_cores": (cpu_auto_plan or {}).get("total_physical_cores")
                                                 or ((cpu_auto_plan or {}).get("physical_cores") if cpu_auto_plan else None),
            "cpu_parallel_last_wave_items": (cpu_auto_plan or {}).get("last_wave_items") if cpu_auto_plan else None,
            "cpu_batch_plan": dict(cpu_batch_plan) if isinstance(cpu_batch_plan, dict) else None,
            "allow_cpu_training": bool(getattr(args, "allow_cpu_training", False)),
            "cpu_training_justification": (getattr(args, "cpu_training_justification", "") or "").strip(),
            "allow_remote_large_data": bool(getattr(args, "allow_remote_large_data", False)),
            "test_log_path": os.path.expanduser(getattr(args, "test_log", "") or "") or None,
            "test_log_profile_loaded": False,
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
        if getattr(args, "test_log", None):
            task["test_log_profile_loaded"] = _apply_test_log_runtime_profile(
                task, args.test_log)
            if not task["test_log_profile_loaded"]:
                task["last_block_reason"] = (
                    f"test log was provided but no tqdm/progress runtime profile could be parsed: {args.test_log}"
                )
        _seed_pending_eta_from_history({"tasks": [task]})
        state["tasks"].append(task)
        state["next_id"] += 1
        save_state(state)
    sources = []
    if hist.get("vram_mb") and args.vram is None: sources.append("vram=hist")
    elif args.vram is not None: sources.append("vram=explicit")
    if hist.get("ram_mb") and args.ram_mb is None: sources.append("ram=hist")
    elif args.ram_mb is not None: sources.append("ram=explicit")
    if int(getattr(args, "cpu_parallel_items", 0) or 0) > 0 and args.cpu is None:
        sources.append("cpu=auto-workers")
    elif hist.get("cpu_cores") and args.cpu is None: sources.append("cpu=hist")
    elif args.cpu is not None: sources.append("cpu=explicit")
    src = ",".join(sources) or "all-defaults"
    print(f"submitted {task['id']}  cpu={cpu_cores} ram={ram_mb}MB vram={est_vram}MB  prio={args.priority}  ({src})  {args.description[:50]}")
    if task.get("cpu_parallel_items"):
        print(f"  cpu-workers: items={task.get('cpu_parallel_items')} "
              f"physical={task.get('cpu_parallel_physical_cores')} "
              f"waves={task.get('cpu_parallel_waves')} "
              f"workers={task.get('cpu_auto_workers')} "
              f"last_wave={task.get('cpu_parallel_last_wave_items')}")
    print(f"  run `dispatch` to launch (resource-aware: respects 1/3 VRAM rule, CPU/RAM headroom).")


def _parse_cpu_node_list(text: Optional[str]) -> list:
    if not text:
        return _cpu_labor_node_names()
    names = [x.strip() for x in str(text).split(",") if x.strip()]
    bad = [n for n in names if n not in NODES]
    if bad:
        sys.exit(f"unknown node(s): {', '.join(bad)}")
    return names


def _cpu_plan_live_node_states(node_names: list) -> dict:
    """Probe current node state for CPU planning; returns {} on probe failure."""
    want = set(node_names or [])
    try:
        nodes = probe_all()
    except Exception:
        return {}
    return {n.get("name"): n for n in nodes
            if n.get("name") in want}


def _print_cpu_plan(plan: list, total_items: int) -> None:
    total_free = sum(int(p.get("physical_cores") or 0) for p in plan)
    total_capacity = sum(int(p.get("total_physical_cores") or p.get("physical_cores") or 0) for p in plan)
    global_waves = max([int(p.get("global_waves") or p.get("waves") or 0) for p in plan] or [0])
    print(f"cpu-plan: total_items={total_items} free_physical={total_free}/{total_capacity} global_waves={global_waves}")
    for p in plan:
        print(f"  {p['node']:11s} range=[{p['start']},{p['end']}) "
              f"items={p['items']} free_physical={p['physical_cores']}/{p.get('total_physical_cores', p['physical_cores'])} "
              f"workers={p['workers']} waves={p['waves']} "
              f"last_wave={p['last_wave_items']}")


def cmd_cpu_plan(args):
    logical_items, item_multiplier, total_items = _cpu_batch_item_counts(args)
    names = _parse_cpu_node_list(getattr(args, "nodes", None))
    node_states = None if getattr(args, "use_total_cores", False) else _cpu_plan_live_node_states(names)
    plan = _cpu_batch_plan(total_items, names, node_states=node_states)
    if args.json:
        print(json.dumps({
            "logical_items": logical_items,
            "item_multiplier": item_multiplier,
            "total_items": total_items,
            "plan": plan,
        }, indent=2))
        return
    if item_multiplier != 1:
        print(f"cpu-plan input: logical_items={logical_items} "
              f"item_multiplier={item_multiplier} total_items={total_items}")
    _print_cpu_plan(plan, total_items)


def cmd_submit_cpu_batch(args):
    logical_items, item_multiplier, total_items = _cpu_batch_item_counts(args)
    names = _parse_cpu_node_list(getattr(args, "nodes", None))
    node_states = None if getattr(args, "use_total_cores", False) else _cpu_plan_live_node_states(names)
    plan = _cpu_batch_plan(total_items, names, node_states=node_states)
    if not plan:
        sys.exit("no CPU batch plan could be built")
    split_tokens = ("{start}", "{end}", "{node}", "{shard_index}", "{shard_id}", "{num_shards}")
    if len(plan) > 1 and not getattr(args, "allow_env_only_shard", False):
        templated = " ".join(str(x or "") for x in (
            args.cmd_template, getattr(args, "result_dir_template", None),
            getattr(args, "local_result_dir_template", None),
        ))
        if not any(tok in templated for tok in split_tokens):
            sys.exit("REFUSED: submit-cpu-batch spans multiple nodes but the command/result templates "
                     "do not contain shard placeholders like {start}/{end}/{node}. "
                     "Pass --allow-env-only-shard only if the script reads SCHEDULEURM_CPU_* env vars.")
    if args.json or args.dry_run:
        rows = []
        for p in plan:
            vals = _cpu_parallel_template_values(
                p, total_items, logical_items, item_multiplier)
            rows.append({
                **p,
                "cmd": _rewrite_cpu_parallel_cmd(
                    args.cmd_template, p, total_items, logical_items, item_multiplier),
                "cwd": _format_cpu_parallel_template(
                    args.cwd, p, total_items, logical_items, item_multiplier),
                "signature": _format_cpu_parallel_template(
                    args.signature, p, total_items, logical_items, item_multiplier),
                "description": _format_cpu_parallel_template(
                    args.description, p, total_items, logical_items, item_multiplier),
                "env": vals,
            })
        if args.json:
            print(json.dumps({
                "logical_items": logical_items,
                "item_multiplier": item_multiplier,
                "total_items": total_items,
                "plan": rows,
            }, indent=2))
        else:
            if item_multiplier != 1:
                print(f"cpu-plan input: logical_items={logical_items} "
                      f"item_multiplier={item_multiplier} total_items={total_items}")
            _print_cpu_plan(plan, total_items)
            for r in rows:
                print(f"  cmd[{r['node']}]: {r['cmd']}")
        if args.dry_run:
            return

    plan_payload = _cpu_batch_log_payload(
        total_items, plan, node_states=node_states,
        use_total_cores=bool(getattr(args, "use_total_cores", False)),
        logical_items=logical_items,
        item_multiplier=item_multiplier,
        templates={
            "cmd": args.cmd_template,
            "cwd": args.cwd,
            "signature": args.signature,
            "description": args.description,
            "result_dir": getattr(args, "result_dir_template", None),
            "local_result_dir": getattr(args, "local_result_dir_template", None),
        },
    )
    notify("cpu_batch_plan", plan_payload, feishu_enabled=False)

    for p in plan:
        vals = _cpu_parallel_template_values(
            p, total_items, logical_items, item_multiplier)
        env = list(getattr(args, "env", None) or [])
        env.extend([
            f"SCHEDULEURM_CPU_TOTAL_ITEMS={total_items}",
            f"SCHEDULEURM_CPU_TOTAL_WORK_ITEMS={total_items}",
            f"SCHEDULEURM_CPU_LOGICAL_ITEMS={logical_items}",
            f"SCHEDULEURM_CPU_ITEM_MULTIPLIER={item_multiplier}",
            f"SCHEDULEURM_CPU_ITEMS_PER_UNIT={item_multiplier}",
            f"SCHEDULEURM_CPU_EPISODES_PER_ITEM={item_multiplier}",
            f"SCHEDULEURM_CPU_SHARD_START={p['start']}",
            f"SCHEDULEURM_CPU_SHARD_END={p['end']}",
            f"SCHEDULEURM_CPU_SHARD_ITEMS={p['items']}",
            f"SCHEDULEURM_CPU_WORKERS={p['workers']}",
            f"SCHEDULEURM_CPU_WAVES={p['waves']}",
            f"SCHEDULEURM_CPU_PHYSICAL_CORES={p['physical_cores']}",
            f"SCHEDULEURM_CPU_TOTAL_PHYSICAL_CORES={p.get('total_physical_cores', p['physical_cores'])}",
            f"SCHEDULEURM_CPU_LAST_WAVE_ITEMS={p['last_wave_items']}",
            f"SCHEDULEURM_CPU_SHARD_INDEX={p['shard_index']}",
            f"SCHEDULEURM_CPU_NUM_SHARDS={p['num_shards']}",
        ])
        ns = argparse.Namespace(
            description=_format_cpu_parallel_template(
                args.description, p, total_items, logical_items, item_multiplier),
            cmd=_rewrite_cpu_parallel_cmd(
                args.cmd_template, p, total_items, logical_items, item_multiplier),
            cwd=_format_cpu_parallel_template(
                args.cwd, p, total_items, logical_items, item_multiplier),
            signature=_format_cpu_parallel_template(
                args.signature, p, total_items, logical_items, item_multiplier),
            vram=0,
            ram_mb=args.ram_mb,
            cpu=int(p["workers"]),
            priority=args.priority,
            project=args.project,
            preferred_node=p["node"],
            require_node=None,
            allowed_nodes=list(names),
            stage_excludes=list(getattr(args, "stage_exclude", None) or []),
            reroute_on_node_down=True,
            node_down_requeue_s=int(getattr(args, "node_down_requeue_s", 0) or 0),
            git_repo=None,
            ckpt_dir=None,
            result_dir=_format_cpu_parallel_template(
                getattr(args, "result_dir_template", None),
                p, total_items, logical_items, item_multiplier),
            local_result_dir=_format_cpu_parallel_template(
                getattr(args, "local_result_dir_template", None),
                p, total_items, logical_items, item_multiplier),
            wait_for_files=[
                _format_cpu_parallel_template(
                    w, p, total_items, logical_items, item_multiplier)
                for w in (getattr(args, "wait_for_file_template", None) or [])
            ],
            test_log=None,
            test_peak_vram_mb=0,
            test_peak_ram_mb=0,
            test_cpu=0,
            ckpt_glob="*",
            resume_flag="",
            env=env,
            allow_cpu_training=bool(getattr(args, "allow_cpu_training", False)),
            cpu_training_justification=getattr(args, "cpu_training_justification", "") or "",
            allow_no_ckpt=bool(getattr(args, "allow_no_ckpt", False)),
            allow_no_resume=bool(getattr(args, "allow_no_resume", False)),
            env_spec=getattr(args, "env_spec", None) or "none",
            image=getattr(args, "image", None) or "",
            allow_shared_ckpt_dir=False,
            allow_shared_result_dir=bool(getattr(args, "allow_shared_result_dir", False)),
            allow_remote_large_data=bool(getattr(args, "allow_remote_large_data", False)),
            allow_duplicate=bool(getattr(args, "allow_duplicate", False)),
            slurm_partition="",
            slurm_account="",
            slurm_qos="",
            cpu_parallel_items=int(p["items"]),
            cpu_parallel_total_items=int(total_items),
            cpu_parallel_logical_items=int(logical_items),
            cpu_parallel_item_multiplier=int(item_multiplier),
            cpu_parallel_start=int(p["start"]),
            cpu_parallel_end=int(p["end"]),
            cpu_parallel_shard_index=int(p["shard_index"]),
            cpu_parallel_num_shards=int(p["num_shards"]),
            cpu_batch_plan=dict(p),
        )
        notify("cpu_batch_shard_submit", {
            "node": p["node"],
            "range": [int(p["start"]), int(p["end"])],
            "items": int(p["items"]),
            "workers": int(p["workers"]),
            "waves": int(p["waves"]),
            "last_wave_items": int(p["last_wave_items"]),
            "signature": ns.signature,
            "description": ns.description,
            "cmd": ns.cmd,
            "env": vals,
            "logical_items": logical_items,
            "item_multiplier": item_multiplier,
            "total_items": total_items,
        }, feishu_enabled=False)
        cmd_submit(ns)


def _project_from_path(path):
    """Heuristic: project name = last path component. Strips trailing /; returns '' if path is None/empty."""
    if not path: return ""
    return os.path.basename(path.rstrip("/")) or ""

def _project_from_pid(node, pid):
    """Read /proc/<pid>/cwd on the node via readlink — gives the actual working dir of the running process."""
    if _node_is_windows(node):
        return ""
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

def _apply_node_cmd_rewrites(node, cmd):
    """Apply deterministic node-local command rewrites before launch."""
    if not node or not cmd:
        return cmd
    for old, new in (NODES.get(node, {}).get("cmd_rewrites") or []):
        old = str(old or "")
        new = str(new or "")
        if old and new and old in cmd:
            cmd = cmd.replace(old, new)
    return cmd

def _task_needs_jax_launch_env(task: dict) -> bool:
    cmd = task.get("cmd") or ""
    project = task.get("project") or ""
    is_gpu = int(task.get("est_vram_mb", DEFAULT_VRAM_MB) or 0) > 0 and not _task_launch_cpu_mode(task)
    return bool(is_gpu and (
        project == "RE-SAC"
        or "jax_experiments" in cmd
        or "jax" in cmd.lower()
    ))


def _launch_extra_env(task):
    """Launch-time env after applying conservative defaults for known JAX GPU tasks."""
    merged = {}
    if _task_needs_jax_launch_env(task):
        # Prevent JAX from preallocating most of a 12GB card and poisoning peak history.
        merged["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        merged["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"
        if task.get("project") == "RE-SAC" and task.get("cwd"):
            merged["PYTHONPATH"] = task["cwd"]
    merged.update(task.get("extra_env") or {})
    return merged

def _node_launch_extra_env(task):
    """Merge node-local launch env with task/JAX defaults."""
    node = task.get("node") or ""
    node_env = dict((NODES.get(node, {}) or {}).get("launch_extra_env") or {})
    merged = node_env if _task_needs_jax_launch_env(task) else {}
    merged.update(_launch_extra_env(task))
    if node:
        merged = {
            k: _rewrite_command_paths_for_node(node, str(v))
            for k, v in merged.items()
        }
    return merged


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

LAUNCH_STAGING_MAX_CANDIDATES_PER_PASS = int(os.environ.get(
    "SCHEDULEURM_LAUNCH_STAGING_MAX_CANDIDATES_PER_PASS", "4"))
# Cold-start staging can include multi-GB Windows CPU workspaces. Keep each
# dispatch/watch pass bounded so a low-urgency sync cannot delay high-priority
# GPU crash recovery for minutes.


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
    if NODES.get(target_node, {}).get("skip_launch_staging"):
        return "ready"
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


def _tar_exclude_args(extra_excludes: Optional[list] = None) -> list:
    patterns = [
        ".git", "__pycache__", "*.pyc",
        "results", "results_*",
        "logs", "logs_*",
        "experiment_output",
        "archive*", "*.tar.gz",
    ]
    for ex in extra_excludes or []:
        clean = str(ex or "").strip().strip("/")
        if not clean:
            continue
        patterns.append(clean)
        patterns.append(f"./{clean}")
    return [f"--exclude={p}" for p in patterns]


def _stage_local_dir_to_windows(local_dir: str, target_node: str, target_dir: str,
                                extra_excludes: Optional[list] = None,
                                timeout_s: int = 600) -> tuple:
    """Stream a local directory to a Windows node using tar over SSH.

    Linux remotes use rsync. Windows OpenSSH hosts usually do not have rsync,
    so we stream a tar archive into PowerShell and extract with Windows' tar.exe.
    This is additive (no --delete equivalent) to avoid deleting remote result or
    checkpoint dirs; launch staging still excludes those paths from the archive.
    """
    if not _node_is_windows(target_node):
        return False, f"target {target_node} is not a Windows node"
    src = Path(local_dir)
    if not src.is_dir():
        return False, f"local directory missing for Windows staging: {local_dir}"
    win_dest = _windows_path_for_node(target_node, target_dir)
    ps = (
        f"$dest={_ps_quote(win_dest)}; "
        "[IO.Directory]::CreateDirectory($dest) | Out-Null; "
        "tar -xf - -C $dest; "
        "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; "
        "Write-Output 'READY'"
    )
    encoded = base64.b64encode(ps.encode("utf-16le")).decode("ascii")
    tar_args = ["tar", "-h", "-C", str(src)] + _tar_exclude_args(extra_excludes) + ["-cf", "-", "."]
    ssh_args = _ssh_base_args(target_node) + [
        "powershell",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy", "Bypass",
        "-EncodedCommand", encoded,
    ]
    tar_proc = None
    try:
        tar_proc = subprocess.Popen(
            tar_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        ssh_proc = subprocess.run(
            ssh_args,
            stdin=tar_proc.stdout,
            capture_output=True,
            timeout=timeout_s,
        )
        if tar_proc.stdout:
            tar_proc.stdout.close()
        _, tar_err = tar_proc.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        if tar_proc:
            try:
                tar_proc.kill()
            except Exception:
                pass
        return False, f"Windows tar staging timeout (>{timeout_s}s)"
    except Exception as e:
        if tar_proc:
            try:
                tar_proc.kill()
            except Exception:
                pass
        return False, f"Windows tar staging exception: {str(e)[:200]}"

    tar_rc = tar_proc.returncode if tar_proc else 1
    tar_err_s = (tar_err or b"").decode("utf-8", "replace").strip()
    if tar_rc != 0:
        return False, f"local tar failed rc={tar_rc}: {tar_err_s[:200]}"
    out_s = (ssh_proc.stdout or b"").decode("utf-8", "replace")
    err_s = (ssh_proc.stderr or b"").decode("utf-8", "replace")
    if ssh_proc.returncode != 0 or "READY" not in out_s:
        return False, (
            f"Windows extract failed rc={ssh_proc.returncode}: "
            f"{(err_s or out_s).strip()[:200]}"
        )
    rc, out, err = _run_windows_ps(
        target_node,
        f"if (Test-Path -LiteralPath {_ps_quote(win_dest)}) {{ 'OK' }} else {{ 'MISSING' }}",
        timeout=10,
        check=False,
    )
    if rc != 0 or "OK" not in out:
        return False, f"Windows staged directory not verified: {win_dest}; {(err or out).strip()[:160]}"
    return True, f"synced to Windows {win_dest}"


def _resume_ckpt_stage_key(source_node: str, target_node: str, ckpt_dir: str) -> tuple:
    return ("resume_ckpt", source_node, target_node, ckpt_dir)


def _resume_location_for_target(source_loc: dict, target_node: str) -> dict:
    staged = dict(source_loc or {})
    staged["node"] = target_node
    if staged.get("path"):
        staged["path"] = _remote_path_for_node(target_node, staged["path"])
    return staged


def _resume_stage_source_for_target(task: dict, target_node: str,
                                    locations: Optional[list] = None) -> Optional[dict]:
    """Pick a checkpoint source that can be rsync'd to target for launch staging.

    rsync is only supported when one side is local. Prefer local as the source
    for remote targets because launch cwd staging already treats local as the
    source of truth. Remote->remote is deliberately unsupported here; without a
    shared FS/tunnel it can silently become "remote starts from zero".
    """
    locations = list(locations if locations is not None else (task.get("resume_locations") or []))
    if not target_node or not locations:
        return None
    for loc in locations:
        if loc.get("node") == target_node:
            return dict(loc)

    target_host = NODES.get(target_node, {}).get("host")
    local_locs = [loc for loc in locations if loc.get("node") == "local"]
    if target_host and local_locs:
        return dict(local_locs[0])
    if target_host is None:
        for loc in locations:
            src_node = loc.get("node")
            if src_node and NODES.get(src_node, {}).get("host"):
                return dict(loc)
    for loc in locations:
        src_node = loc.get("node")
        if not src_node:
            continue
        src_host = NODES.get(src_node, {}).get("host")
        if bool(src_host) != bool(target_host):
            return dict(loc)
    return None


def _resume_checkpoint_stage_check(task: dict, target_node: str,
                                   locations: Optional[list] = None) -> tuple:
    """Cache-only check for launch-time checkpoint staging.

    Returns (state, source_loc, reason) where state is one of:
      ready        - target already has ckpt OR a fresh rsync cache entry exists
      needs_stage  - outside-lock staging has not completed yet
      cap_exceeded - ckpt_dir is too large to sync automatically
      stage_failed - recent rsync/probe failure for this source/target
      unsupported  - checkpoint exists, but no supported source->target route
    """
    locations = list(locations if locations is not None else (task.get("resume_locations") or []))
    if not locations:
        return "ready", None, "no existing checkpoint"
    for loc in locations:
        if loc.get("node") == target_node:
            return "ready", dict(loc), "checkpoint already on target"
    src = _resume_stage_source_for_target(task, target_node, locations)
    if not src:
        return "unsupported", None, "no local-side rsync route for checkpoint staging"
    ckpt_dir = task.get("ckpt_dir")
    if not ckpt_dir:
        return "ready", src, "no ckpt_dir"
    key = _resume_ckpt_stage_key(src.get("node"), target_node, ckpt_dir)
    if _staging_cache_hit(key):
        staged = _resume_location_for_target(src, target_node)
        return "ready", staged, "checkpoint staged to target"
    ts = _STAGING_CAP_EXCEEDED.get(key)
    if ts is not None:
        if (time.time() - ts) > STAGING_TTL_S:
            _STAGING_CAP_EXCEEDED.pop(key, None)
        else:
            return "cap_exceeded", src, "checkpoint directory exceeds staging cap"
    fail = _STAGING_FAILS.get(key)
    if fail is not None:
        fail_ts = fail[0] if isinstance(fail, tuple) else fail
        if (time.time() - fail_ts) > STAGING_FAIL_COOLDOWN_S:
            _STAGING_FAILS.pop(key, None)
        else:
            msg = fail[1] if isinstance(fail, tuple) and len(fail) > 1 else "unknown"
            return "stage_failed", src, msg
    return "needs_stage", src, "checkpoint not yet staged to target"


def _record_staged_resume_location(task: dict, target_node: str, source_loc: Optional[dict]) -> None:
    """Record that source_loc's checkpoint path is now available on target_node."""
    if not source_loc or not target_node:
        return
    staged_loc = _resume_location_for_target(source_loc, target_node)
    locs = [loc for loc in (task.get("resume_locations") or [])
            if loc.get("node") != target_node]
    locs.append(staged_loc)
    locs.sort(key=lambda x: (float(x.get("mtime") or 0), str(x.get("node") or "")), reverse=True)
    task["resume_locations"] = locs
    ordered_nodes = []
    for loc in locs:
        node = loc.get("node")
        if node and node not in ordered_nodes:
            ordered_nodes.append(node)
    task["resume_preferred_nodes"] = ordered_nodes
    best = locs[0] if locs else staged_loc
    task["resume_checkpoint_node"] = best.get("node")
    task["resume_from"] = (next((loc.get("path") for loc in locs
                                 if loc.get("node") == target_node and loc.get("path")), None)
                           or best.get("path")
                           or task.get("resume_from"))


def _stage_resume_ckpt_for_launch(task: dict, target_node: str,
                                  source_loc: Optional[dict] = None,
                                  max_ckpt_mb: int = MIGRATION_MAX_CKPT_SIZE_MB) -> tuple:
    """Pre-launch rsync of a resume checkpoint directory to a target node.

    This fills the gap between migration staging and first launch staging:
    queued resume tasks with a small checkpoint on local should be able to run
    on an idle remote node after rsync, while very large checkpoint trees (for
    example SimpleSAC snapshot data) stay pinned to the checkpoint-local node.
    """
    if not _task_requires_resume_scan(task):
        return True, "not a resume-scanned task"
    ckpt_dir = task.get("ckpt_dir")
    if not ckpt_dir:
        return True, "no ckpt_dir"
    if not target_node:
        return False, "no target node"
    target_host = NODES.get(target_node, {}).get("host")
    if source_loc is None:
        state, source_loc, msg = _resume_checkpoint_stage_check(task, target_node)
        if state == "ready":
            return True, msg
        if not source_loc:
            return False, msg
    source_node = source_loc.get("node")
    if not source_node:
        return False, "resume source has no node"
    if source_node == target_node:
        return True, "source==target; checkpoint already local"
    src_host = NODES.get(source_node, {}).get("host")
    if _node_is_windows(source_node):
        return False, (
            f"ckpt_dir {ckpt_dir} staging from Windows source {source_node} "
            "is not supported yet; keep resume task on that Windows node or "
            "copy the checkpoint to local first"
        )
    if src_host and target_host:
        return False, (
            f"ckpt_dir {ckpt_dir} needs rsync but remote→remote is unsupported "
            f"(src={src_host}, tgt={target_host})"
        )

    key = _resume_ckpt_stage_key(source_node, target_node, ckpt_dir)
    if _staging_cache_hit(key):
        return True, "cache hit (checkpoint already staged)"

    source_ckpt_dir = _remote_path_for_node(source_node, ckpt_dir)
    target_ckpt_dir = _remote_path_for_node(target_node, ckpt_dir)
    try:
        rc_test, _, _ = run_on(source_node, f"test -d {shlex.quote(source_ckpt_dir)}",
                               timeout=5, check=False)
    except Exception as e:
        return False, f"ckpt_dir reachability check on {source_node} failed: {str(e)[:120]}"
    if rc_test != 0:
        return False, f"ckpt_dir {source_ckpt_dir} missing on resume source {source_node}"

    size_mb = -1
    try:
        rc, out, _ = run_on(
            source_node,
            f"du -sm {shlex.quote(source_ckpt_dir)} 2>/dev/null | awk '{{print $1}}'",
            timeout=15, check=False,
        )
        if rc == 0 and out.strip().isdigit():
            size_mb = int(out.strip())
    except Exception:
        pass
    if size_mb < 0:
        return False, f"ckpt_dir {ckpt_dir} exists on {source_node} but size probe failed"
    if size_mb > max_ckpt_mb:
        _STAGING_CAP_EXCEEDED[key] = time.time()
        return False, (
            f"CAP_EXCEEDED: ckpt_dir {ckpt_dir} is {size_mb}MB > "
            f"{max_ckpt_mb}MB cap; keep task on checkpoint-local node"
        )

    if _node_is_windows(target_node):
        if source_node != "local" or src_host:
            return False, (
                f"ckpt_dir {ckpt_dir} needs Windows staging, but only "
                f"local→Windows is supported (source={source_node})"
            )
        ok, msg = _stage_local_dir_to_windows(
            ckpt_dir,
            target_node,
            ckpt_dir,
            timeout_s=600,
        )
        if not ok:
            return False, msg
        resume_path = source_loc.get("path")
        try:
            if resume_path:
                win_resume = _windows_path_for_node(target_node, resume_path)
                rc2, out2, _ = _run_windows_ps(
                    target_node,
                    f"if (Test-Path -LiteralPath {_ps_quote(win_resume)}) {{ 'OK' }} else {{ 'MISSING' }}",
                    timeout=10,
                    check=False,
                )
            else:
                win_dir = _windows_path_for_node(target_node, ckpt_dir)
                rc2, out2, _ = _run_windows_ps(
                    target_node,
                    f"$x=Get-ChildItem -LiteralPath {_ps_quote(win_dir)} -File -ErrorAction SilentlyContinue | Select-Object -First 1; if ($x) {{ 'OK' }} else {{ 'MISSING' }}",
                    timeout=10,
                    check=False,
                )
            if rc2 != 0 or "OK" not in out2:
                return False, f"ckpt_dir {ckpt_dir} not verified on {target_node} after Windows staging"
        except Exception as e:
            return False, f"post-Windows ckpt check failed: {str(e)[:120]}"
        _STAGING_CACHE[key] = time.time()
        _STAGING_CAP_EXCEEDED.pop(key, None)
        _STAGING_FAILS.pop(key, None)
        return True, f"synced ckpt_dir to Windows ({size_mb}MB): {msg}"

    relay_node = _relay_node_for_node(target_node)
    if relay_node and not src_host:
        relay_ckpt_dir = _relay_path_for_node(target_node, target_ckpt_dir)
        try:
            run_on(relay_node, f"mkdir -p {shlex.quote(relay_ckpt_dir)}", timeout=10, check=False)
        except Exception:
            pass
        src_path = source_ckpt_dir.rstrip("/") + "/"
        relay_dst = _rsync_path_for_node(relay_node, relay_ckpt_dir.rstrip("/") + "/")
        try:
            rsync_args = ["rsync", "-az", "--partial"]
            relay_shell = _ssh_rsync_shell_for_node(relay_node)
            if relay_shell:
                rsync_args.extend(["-e", relay_shell])
            rsync_args.extend([src_path, relay_dst])
            r = subprocess.run(rsync_args, capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                return False, f"rsync ckpt local→relay rc={r.returncode}: {(r.stderr or '').strip()[:200]}"
        except subprocess.TimeoutExpired:
            return False, "rsync ckpt local→relay timeout (>600s)"
        except Exception as e:
            return False, f"rsync ckpt local→relay exception: {str(e)[:200]}"
        try:
            run_on(target_node, f"mkdir -p {shlex.quote(target_ckpt_dir)}", timeout=10, check=False)
        except Exception:
            pass
        target = _relay_ssh_target_for_node(target_node)
        relay_cmd = " ".join(shlex.quote(a) for a in [
            "rsync", "-az", "--partial",
            relay_ckpt_dir.rstrip("/") + "/",
            f"{target}:{target_ckpt_dir.rstrip('/')}/",
        ])
        try:
            rc, out, err = run_on(relay_node, relay_cmd, timeout=600, check=False)
            if rc != 0:
                return False, f"rsync ckpt relay→target rc={rc}: {(err or out).strip()[:200]}"
        except Exception as e:
            return False, f"rsync ckpt relay→target exception: {str(e)[:200]}"
        try:
            if resume_path := source_loc.get("path"):
                target_resume = _remote_path_for_node(target_node, resume_path)
                rc2, _, _ = run_on(target_node, f"test -e {shlex.quote(target_resume)}",
                                   timeout=5, check=False)
            else:
                rc2, _, _ = run_on(
                    target_node,
                    f"ls -1 {shlex.quote(target_ckpt_dir)} 2>/dev/null | head -1",
                    timeout=5, check=False,
                )
            if rc2 != 0:
                return False, f"ckpt_dir {target_ckpt_dir} not verified on {target_node} after relay rsync"
        except Exception as e:
            return False, f"post-relay ckpt check failed: {str(e)[:120]}"
        _STAGING_CACHE[key] = time.time()
        _STAGING_CAP_EXCEEDED.pop(key, None)
        _STAGING_FAILS.pop(key, None)
        return True, f"synced ckpt_dir via {relay_node} ({size_mb}MB)"

    try:
        run_on(target_node, f"mkdir -p {shlex.quote(target_ckpt_dir)}", timeout=10, check=False)
    except Exception:
        pass

    src_path = _rsync_path_for_node(source_node, source_ckpt_dir.rstrip("/") + "/")
    dst_path = _rsync_path_for_node(target_node, target_ckpt_dir.rstrip("/") + "/")
    try:
        rsync_args = ["rsync", "-az", "--partial"]
        rsync_shell = _rsync_shell_for_pair(source_node, target_node)
        if rsync_shell:
            rsync_args.extend(["-e", rsync_shell])
        rsync_args.extend([src_path, dst_path])
        r = subprocess.run(rsync_args, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            return False, f"rsync ckpt failed rc={r.returncode}: {(r.stderr or '').strip()[:200]}"
    except subprocess.TimeoutExpired:
        return False, "rsync ckpt timeout (>600s)"
    except Exception as e:
        return False, f"rsync ckpt exception: {str(e)[:200]}"

    resume_path = source_loc.get("path")
    try:
        if resume_path:
            target_resume = _remote_path_for_node(target_node, resume_path)
            rc2, _, _ = run_on(target_node, f"test -e {shlex.quote(target_resume)}",
                               timeout=5, check=False)
        else:
            rc2, _, _ = run_on(
                target_node,
                f"ls -1 {shlex.quote(target_ckpt_dir)} 2>/dev/null | head -1",
                timeout=5, check=False,
            )
        if rc2 != 0:
            return False, f"ckpt_dir {target_ckpt_dir} not verified on {target_node} after rsync"
    except Exception as e:
        return False, f"post-rsync ckpt check failed: {str(e)[:120]}"

    _STAGING_CACHE[key] = time.time()
    _STAGING_CAP_EXCEEDED.pop(key, None)
    _STAGING_FAILS.pop(key, None)
    return True, f"synced ckpt_dir ({size_mb}MB)"


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
    cwd_key = ("local", target_node, cwd)
    if _staging_cache_hit(cwd_key):
        return (True, "cache hit (already synced within TTL)")
    if _node_is_windows(target_node) and _is_windows_native_path(cwd):
        win_cwd = _windows_path_for_node(target_node, cwd)
        try:
            rc, out, err = _run_windows_ps(
                target_node,
                f"if (Test-Path -LiteralPath {_ps_quote(win_cwd)} -PathType Container) {{ 'OK' }} else {{ 'MISSING' }}",
                timeout=10,
                check=False,
            )
        except Exception as e:
            return (False, f"Windows target cwd probe failed on {target_node}: {str(e)[:160]}")
        if rc == 0 and "OK" in (out or ""):
            _STAGING_CACHE[cwd_key] = time.time()
            _STAGING_CAP_EXCEEDED.pop(cwd_key, None)
            _STAGING_FAILS.pop(cwd_key, None)
            return (True, f"target-native Windows cwd exists on {target_node}: {win_cwd}")
        return (False, f"Windows target cwd missing on {target_node}: {win_cwd}; {(err or out or '').strip()[:120]}")
    # Source must be local; if cwd doesn't exist locally we can't be the
    # source-of-truth, so don't try to rsync from nothing.
    if not Path(cwd).exists():
        return (False,
                f"cwd {cwd} does not exist on local; can't seed target {target_node}")

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

    if _node_is_windows(target_node):
        ok, msg = _stage_local_dir_to_windows(
            cwd,
            target_node,
            cwd,
            extra_excludes=extra_excludes,
            timeout_s=600,
        )
        if not ok:
            return (False, msg)
        _STAGING_CACHE[cwd_key] = time.time()
        _STAGING_CAP_EXCEEDED.pop(cwd_key, None)
        _STAGING_FAILS.pop(cwd_key, None)
        return (True, f"{msg} ({cwd_size_mb}MB)")

    # Ensure target dir exists (mkdir -p is idempotent; cheap).
    remote_cwd = _remote_path_for_node(target_node, cwd)
    try:
        run_on(target_node, f"mkdir -p {shlex.quote(remote_cwd)}", timeout=10, check=False)
    except Exception:
        pass  # if mkdir fails the rsync below will too — surface the rsync error

    relay_node = _relay_node_for_node(target_node)
    if relay_node:
        relay_cwd = _relay_path_for_node(target_node, remote_cwd)
        try:
            run_on(relay_node, f"mkdir -p {shlex.quote(relay_cwd)}", timeout=10, check=False)
        except Exception:
            pass

        src_path = cwd.rstrip("/") + "/"
        relay_dst = _rsync_path_for_node(relay_node, relay_cwd.rstrip("/") + "/")
        local_to_relay = ["rsync", "-az", "--partial", "--delete",
                          "--exclude=.git/", "--exclude=__pycache__/", "--exclude=*.pyc",
                          "--exclude=results/", "--exclude=results_*/",
                          "--exclude=logs/", "--exclude=logs_*/",
                          "--exclude=experiment_output/",
                          "--exclude=archive*/", "--exclude=*.tar.gz"]
        if extra_excludes:
            for ex in extra_excludes:
                if ex:
                    local_to_relay.extend([f"--exclude={ex}"])
        relay_shell = _ssh_rsync_shell_for_node(relay_node)
        if relay_shell:
            local_to_relay.extend(["-e", relay_shell])
        local_to_relay.extend([src_path, relay_dst])
        try:
            r = subprocess.run(local_to_relay, capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                return (False, f"rsync local→relay rc={r.returncode}: {(r.stderr or '').strip()[:200]}")
        except subprocess.TimeoutExpired:
            return (False, "rsync local→relay timeout (>600s)")
        except Exception as e:
            return (False, f"rsync local→relay exception: {str(e)[:200]}")

        try:
            run_on(target_node, f"mkdir -p {shlex.quote(remote_cwd)}", timeout=10, check=False)
        except Exception:
            pass
        target = _relay_ssh_target_for_node(target_node)
        relay_src = relay_cwd.rstrip("/") + "/"
        relay_to_hpc = ["rsync", "-az", "--partial", "--delete",
                        "--exclude=.git/", "--exclude=__pycache__/", "--exclude=*.pyc",
                        "--exclude=results/", "--exclude=results_*/",
                        "--exclude=logs/", "--exclude=logs_*/",
                        "--exclude=experiment_output/",
                        "--exclude=archive*/", "--exclude=*.tar.gz"]
        if extra_excludes:
            for ex in extra_excludes:
                if ex:
                    relay_to_hpc.extend([f"--exclude={ex}"])
        relay_to_hpc.extend([relay_src, f"{target}:{remote_cwd.rstrip('/')}/"])
        relay_cmd = " ".join(shlex.quote(a) for a in relay_to_hpc)
        try:
            rc, out, err = run_on(relay_node, relay_cmd, timeout=600, check=False)
            if rc != 0:
                return (False, f"rsync relay→target rc={rc}: {(err or out).strip()[:200]}")
        except Exception as e:
            return (False, f"rsync relay→target exception: {str(e)[:200]}")

        _STAGING_CACHE[cwd_key] = time.time()
        _STAGING_CAP_EXCEEDED.pop(cwd_key, None)
        _STAGING_FAILS.pop(cwd_key, None)
        return (True, f"synced via {relay_node} ({cwd_size_mb}MB)")

    src_path = cwd.rstrip("/") + "/"
    dst_path = _rsync_path_for_node(target_node, remote_cwd.rstrip("/") + "/")
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
    rsync_shell = _ssh_rsync_shell_for_node(target_node)
    if rsync_shell:
        rsync_args.extend(["-e", rsync_shell])
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
            for ex in (t.get("stage_excludes") or []):
                ex = str(ex or "").strip().strip("/")
                if ex:
                    protected_under_cwd.setdefault(t_cwd_norm, set()).add(ex)
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
        queued_tasks = [t for t in state.get("tasks", []) if t.get("status") == "queued"]
        high_gpu_tasks = [
            t for t in queued_tasks
            if t.get("priority") == "high" and int(t.get("est_vram_mb") or 0) > 0
        ]
        if high_gpu_tasks:
            # Fast lane for crash/reboot recovery: do not let unrelated CPU
            # workspace syncs delay high-priority GPU retries.
            queued_tasks = high_gpu_tasks
        gpu_staging_pending = any(
            int(t.get("est_vram_mb") or 0) > 0
            and not _queued_wait_for_file_block_reason(t)
            for t in queued_tasks
        )

        candidates: set = set()
        ckpt_candidates: list = []
        for t in queued_tasks:
            if _queued_wait_for_file_block_reason(t):
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
            allowed = set(t.get("allowed_nodes") or [])
            if allowed:
                tgts = [tn for tn in tgts if tn in allowed]
            for tn in tgts:
                node_info = NODES.get(tn, {})
                if node_info.get("host") is None:
                    continue
                if node_info.get("skip_launch_staging"):
                    continue
                if gpu_staging_pending and _node_is_windows(tn):
                    continue
                if node_info.get("stage_only_when_targeted"):
                    targeted = (
                        require == tn
                        or t.get("preferred_node") == tn
                        or tn in allowed
                        or _task_requests_slurm(t)
                    )
                    if not targeted:
                        continue
                cpu_fallback_target = (
                    int(t.get("est_vram_mb") or 0) > 0
                    and (_node_is_windows(tn) or node_info.get("max_vram_per_task") == 0)
                )
                if cpu_fallback_target and _node_cpu_fallback_block_reason(t, tn, node_info):
                    continue
                cwd_key = ("local", tn, cwd)
                if _staging_cache_hit(cwd_key):
                    continue
                cap_ts = _STAGING_CAP_EXCEEDED.get(cwd_key)
                if cap_ts is not None and (time.time() - cap_ts) <= STAGING_TTL_S:
                    continue
                candidates.add((tn, cwd))
            if _task_requires_resume_scan(t):
                locs = list(t.get("resume_locations") or [])
                if locs:
                    for tn in tgts:
                        node_info = NODES.get(tn, {})
                        if node_info.get("host") is None:
                            continue
                        if node_info.get("skip_launch_staging"):
                            continue
                        if gpu_staging_pending and _node_is_windows(tn):
                            continue
                        if node_info.get("stage_only_when_targeted"):
                            targeted = (
                                require == tn
                                or t.get("preferred_node") == tn
                                or tn in allowed
                                or _task_requests_slurm(t)
                            )
                            if not targeted:
                                continue
                        cpu_fallback_target = (
                            int(t.get("est_vram_mb") or 0) > 0
                            and (_node_is_windows(tn) or node_info.get("max_vram_per_task") == 0)
                        )
                        if cpu_fallback_target and _node_cpu_fallback_block_reason(t, tn, node_info):
                            continue
                        state_ckpt, src_loc, _ = _resume_checkpoint_stage_check(t, tn, locs)
                        if state_ckpt != "needs_stage" or not src_loc:
                            continue
                        ckpt_candidates.append((dict(t), tn, dict(src_loc)))
    except Exception as e:
        try:
            notify("launch_staging_snapshot_error", {"error": str(e)[:200]},
                   feishu_enabled=False)
        except Exception:
            pass
        return

    if not candidates and not ckpt_candidates:
        return

    def _launch_stage_candidate_key(item):
        tn, cwd = item
        node_info = NODES.get(tn, {})
        caps = set(str(c).lower() for c in (node_info.get("capabilities") or []))
        if _node_is_windows(tn):
            node_rank = 3
        elif "cuda" in caps or "jax_cuda" in caps or "torch_cuda" in caps:
            node_rank = 0
        elif _slurm_mode_enabled(node_info.get("slurm_backend")):
            node_rank = 1
        else:
            node_rank = 2
        return (node_rank, tn, cwd)

    ordered_candidates = sorted(candidates, key=_launch_stage_candidate_key)
    if LAUNCH_STAGING_MAX_CANDIDATES_PER_PASS > 0:
        ordered_candidates = ordered_candidates[:LAUNCH_STAGING_MAX_CANDIDATES_PER_PASS]

    for tn, cwd in ordered_candidates:
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

    for task_snapshot, tn, src_loc in ckpt_candidates:
        source_node = src_loc.get("node")
        ckpt_dir = task_snapshot.get("ckpt_dir")
        ckpt_key = _resume_ckpt_stage_key(source_node, tn, ckpt_dir)
        if _staging_cache_hit(ckpt_key):
            continue
        try:
            ok, msg = _stage_resume_ckpt_for_launch(task_snapshot, tn, src_loc)
            if not ok:
                if msg.startswith("CAP_EXCEEDED:"):
                    continue
                _STAGING_FAILS[ckpt_key] = (time.time(), msg[:200])
                try:
                    notify("launch_ckpt_staging_failed",
                           {"task_id": task_snapshot.get("id"),
                            "source": source_node,
                            "target": tn,
                            "ckpt_dir": ckpt_dir,
                            "reason": msg[:200]},
                           feishu_enabled=False)
                except Exception:
                    pass
        except Exception as e:
            _STAGING_FAILS[ckpt_key] = (
                time.time(), f"exception: {str(e)[:150]}")
            try:
                notify("launch_ckpt_staging_exception",
                       {"task_id": task_snapshot.get("id"),
                        "source": source_node,
                        "target": tn,
                        "ckpt_dir": ckpt_dir,
                        "error": str(e)[:200]},
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
        source_cwd = _remote_path_for_node(source_node, cwd)
        target_cwd = _remote_path_for_node(target_node, cwd)
        # Build src/dst rsync paths
        src_path = _rsync_path_for_node(source_node, source_cwd.rstrip("/") + "/")
        dst_path = _rsync_path_for_node(target_node, target_cwd.rstrip("/") + "/")
        mkdir_cmd = f"mkdir -p {shlex.quote(target_cwd)}"
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
                    f"--exclude='*.pyc' {shlex.quote(source_cwd)} 2>/dev/null | "
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
            rsync_args = ["rsync", "-az", "--partial",
                          "--exclude=.git", "--exclude=__pycache__",
                          "--exclude=*.pyc"]
            rsync_shell = _rsync_shell_for_pair(source_node, target_node)
            if rsync_shell:
                rsync_args.extend(["-e", rsync_shell])
            rsync_args.extend([src_path, dst_path])
            r = _sp.run(rsync_args, capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                return (False, f"rsync cwd failed rc={r.returncode}: {r.stderr.strip()[:200]}")
        except _sp.TimeoutExpired:
            return (False, "rsync cwd timeout (>10min)")
        except Exception as e:
            return (False, f"rsync cwd exception: {e}")
        # Verify after rsync
        try:
            rc2, _, _ = run_on(target_node, f"test -d {shlex.quote(target_cwd)}",
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
            source_ckpt_dir = _remote_path_for_node(source_node, ckpt_dir)
            try:
                rc_test, _, _ = run_on(
                    source_node, f"test -d {shlex.quote(source_ckpt_dir)}",
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
                                    f"du -sm {shlex.quote(source_ckpt_dir)} 2>/dev/null | "
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
            source_ckpt_dir = _remote_path_for_node(source_node, ckpt_dir)
            target_ckpt_dir = _remote_path_for_node(target_node, ckpt_dir)
            src_path = _rsync_path_for_node(source_node, source_ckpt_dir.rstrip("/") + "/")
            dst_path = _rsync_path_for_node(target_node, target_ckpt_dir.rstrip("/") + "/")
            try:
                run_on(target_node, f"mkdir -p {shlex.quote(target_ckpt_dir)}",
                       timeout=10, check=False)
                import subprocess as _sp
                rsync_args = ["rsync", "-az", "--partial"]
                rsync_shell = _rsync_shell_for_pair(source_node, target_node)
                if rsync_shell:
                    rsync_args.extend(["-e", rsync_shell])
                rsync_args.extend([src_path, dst_path])
                r = _sp.run(rsync_args, capture_output=True, text=True, timeout=600)
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
                                      f"ls -1 {shlex.quote(target_ckpt_dir)} 2>/dev/null | head -1",
                                      timeout=5, check=False)
                if rc2 != 0 or not (out2 or "").strip():
                    return (False, f"ckpt_dir {target_ckpt_dir} appears empty on {target_node} "
                                   f"after rsync (rc={rc2}); migration aborted")
            except Exception as e:
                return (False, f"post-rsync ckpt check failed: {e}")
            # Only cache on successful + verified rsync
            _STAGING_CACHE[ckpt_key] = time.time()

    # Step 4: env probe — extract python path after node-local rewrites.
    # A task may be submitted with a local-only interpreter path while the target
    # node has an equivalent env under a different absolute path.
    cmd_str = _apply_node_cmd_rewrites(target_node, task.get("cmd") or "")
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


def _ssh_rsync_shell_for_node(node: str) -> str:
    """Return the ssh transport command rsync should use for a configured node."""
    return " ".join(shlex.quote(a) for a in _ssh_base_args(node)[:-1])


def _ssh_target_for_node(node: str) -> str:
    return _ssh_base_args(node)[-1]


def _rsync_path_for_node(node: str, path: str) -> str:
    host = (NODES.get(node, {}) or {}).get("host")
    if not host:
        return path
    return f"{_ssh_target_for_node(node)}:{path}"


def _relay_node_for_node(node: str) -> Optional[str]:
    relay = (NODES.get(node, {}) or {}).get("relay_node")
    return str(relay) if relay else None


def _relay_path_for_node(node: str, remote_path: str) -> str:
    info = NODES.get(node, {}) or {}
    root = str(info.get("relay_root") or f"/tmp/scheduleurm-relay/{node}").rstrip("/")
    clean = os.path.normpath(str(remote_path or "")).lstrip("/")
    if not clean or clean == ".":
        clean = "_root"
    return f"{root}/{clean}"


def _relay_ssh_target_for_node(node: str) -> str:
    info = NODES.get(node, {}) or {}
    explicit = info.get("relay_ssh_target")
    if explicit:
        return str(explicit)
    host = info.get("host")
    user = info.get("ssh_user")
    return f"{user}@{host}" if user and "@" not in str(host) else str(host)


def _rsync_shell_for_pair(source_node: str, target_node: str) -> Optional[str]:
    src_host = (NODES.get(source_node or "", {}) or {}).get("host")
    tgt_host = (NODES.get(target_node or "", {}) or {}).get("host")
    if src_host and tgt_host:
        return None
    if src_host:
        return _ssh_rsync_shell_for_node(source_node)
    if tgt_host:
        return _ssh_rsync_shell_for_node(target_node)
    return None


def _sync_windows_result(candidate: dict) -> tuple:
    """Pull a Windows result directory to local via SSH+tar.

    Windows OpenSSH hosts in this scheduler setup do not expose rsync, and
    result_dir is usually submitted as the Linux workspace path. Map that path
    back to the configured Windows workspace root before archiving.
    """
    node = candidate["node"]
    remote_dir = _windows_path_for_node(node, candidate["result_dir"].rstrip("/"))
    dst = candidate["local_result_dir"].rstrip("/") + "/"
    try:
        Path(dst).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return (False, f"mkdir local target failed: {str(e)[:120]}")

    ps = (
        f"$src={_ps_quote(remote_dir)}; "
        "if (!(Test-Path -LiteralPath $src)) { Write-Error \"missing result_dir $src\"; exit 44 }; "
        "tar -cf - -C $src .; "
        "exit $LASTEXITCODE"
    )
    encoded = base64.b64encode(ps.encode("utf-16le")).decode("ascii")
    ssh_args = _ssh_base_args(node) + [
        "powershell",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy", "Bypass",
        "-EncodedCommand", encoded,
    ]
    ssh_proc = None
    try:
        ssh_proc = subprocess.Popen(ssh_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        tar_proc = subprocess.run(
            ["tar", "-xf", "-", "-C", dst],
            stdin=ssh_proc.stdout,
            capture_output=True,
            text=True,
            timeout=RESULT_SYNC_TIMEOUT_S,
        )
        if ssh_proc.stdout:
            ssh_proc.stdout.close()
        _, ssh_err = ssh_proc.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        if ssh_proc:
            try:
                ssh_proc.kill()
            except Exception:
                pass
        return (False, f"Windows result tar sync timeout (>{RESULT_SYNC_TIMEOUT_S}s)")
    except Exception as e:
        if ssh_proc:
            try:
                ssh_proc.kill()
            except Exception:
                pass
        return (False, f"Windows result tar sync exception: {str(e)[:200]}")

    ssh_err_s = (ssh_err or b"").decode("utf-8", "replace").strip()
    if ssh_proc.returncode != 0:
        return (False, f"Windows result tar ssh rc={ssh_proc.returncode}: {ssh_err_s[:200]}")
    if tar_proc.returncode != 0:
        return (False, f"local result tar extract rc={tar_proc.returncode}: {(tar_proc.stderr or '').strip()[:200]}")
    return (True, "ok")


def _sync_one_result(candidate: dict) -> tuple:
    """Pull remote `result_dir` → local `local_result_dir`.

    Trailing slash on rsync source means "contents", so dst structure mirrors
    source. The caller chooses the result_dir boundary; if it points at a run
    directory, checkpoints under that run are mirrored too.
    """
    node = candidate.get("node") or ""
    if candidate.get("result_dirs"):
        result_dirs = list(candidate.get("result_dirs") or [])
        local_dirs = list(candidate.get("local_result_dirs") or [])
        errors = []
        synced = 0
        for idx, rd in enumerate(result_dirs):
            child = dict(candidate)
            child.pop("result_dirs", None)
            child.pop("local_result_dirs", None)
            child.pop("local_result_dir_base", None)
            child["result_dir"] = rd
            if idx < len(local_dirs) and local_dirs[idx]:
                child["local_result_dir"] = local_dirs[idx]
            elif candidate.get("local_result_dir_base"):
                child["local_result_dir"] = os.path.join(
                    candidate["local_result_dir_base"],
                    os.path.basename(str(rd).rstrip("/")),
                )
            else:
                child["local_result_dir"] = rd
            ok, msg = _sync_one_result(child)
            if ok:
                synced += 1
            else:
                errors.append(f"{rd}: {msg}")
        if errors:
            return (False, f"synced {synced}/{len(result_dirs)} dirs; "
                           + "; ".join(errors)[:220])
        return (True, f"synced {synced} dirs")

    if _node_is_windows(node):
        return _sync_windows_result(candidate)

    remote_dir = _remote_path_for_node(node, candidate["result_dir"].rstrip("/"))
    dst = candidate['local_result_dir'].rstrip('/') + "/"
    try:
        Path(dst).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return (False, f"mkdir local target failed: {str(e)[:120]}")
    relay_node = _relay_node_for_node(node)
    if relay_node:
        relay_dir = _relay_path_for_node(node, remote_dir)
        target = _relay_ssh_target_for_node(node)
        pull_to_relay = [
            "rsync", "-az", "--partial",
            f"{target}:{remote_dir.rstrip('/')}/",
            relay_dir.rstrip("/") + "/",
        ]
        pull_cmd = " ".join(shlex.quote(a) for a in pull_to_relay)
        try:
            run_on(relay_node, f"mkdir -p {shlex.quote(relay_dir)}", timeout=10, check=False)
            rc, out, err = run_on(relay_node, pull_cmd, timeout=RESULT_SYNC_TIMEOUT_S, check=False)
            if rc != 0:
                return (False, f"relay result pull rc={rc}: {(err or out).strip()[:200]}")
        except Exception as e:
            return (False, f"relay result pull exception: {str(e)[:200]}")
        try:
            r = subprocess.run(
                ["rsync", "-az", "--partial", "-e", _ssh_rsync_shell_for_node(relay_node),
                 _rsync_path_for_node(relay_node, relay_dir.rstrip("/") + "/"), dst],
                capture_output=True, text=True, timeout=RESULT_SYNC_TIMEOUT_S,
            )
            if r.returncode != 0:
                return (False, f"relay→local rsync rc={r.returncode}: {r.stderr.strip()[:200]}")
            return (True, "ok")
        except subprocess.TimeoutExpired:
            return (False, f"relay→local rsync timeout (>{RESULT_SYNC_TIMEOUT_S}s)")
        except Exception as e:
            return (False, f"relay→local rsync exception: {str(e)[:200]}")

    src = f"{_ssh_target_for_node(node)}:{remote_dir.rstrip('/')}/"
    try:
        r = subprocess.run(
            ["rsync", "-az", "--partial", "-e", _ssh_rsync_shell_for_node(node), src, dst],
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
RESULT_SYNC_MAX_PER_CYCLE = max(1, int(os.environ.get("SCHEDULEURM_RESULT_SYNC_MAX_PER_CYCLE", "1")))


def _sync_completed_results_outside_lock(max_candidates: Optional[int] = None):
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
    limit = RESULT_SYNC_MAX_PER_CYCLE if max_candidates is None else max(1, int(max_candidates))
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
                if len(candidates) >= limit:
                    break
                if t.get("status") != "done":
                    continue
                rd = t.get("result_dir")
                rds = list(t.get("result_dirs") or [])
                if not rd and not rds:
                    rds = _infer_bapr_result_dirs_from_cmd(t.get("cmd", ""), t.get("cwd", ""))
                if not rd and not rds:
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
                candidate = {
                    "id": t["id"],
                    "node": node,
                    "host": host,
                    "result_dir": rd,
                    "local_result_dir": t.get("local_result_dir") or rd,
                }
                if rds:
                    candidate["result_dirs"] = rds
                    if t.get("local_result_dirs"):
                        candidate["local_result_dirs"] = list(t.get("local_result_dirs") or [])
                    elif t.get("local_result_dir"):
                        candidate["local_result_dir_base"] = t.get("local_result_dir")
                    candidate["result_dir"] = rds[0]
                    candidate["local_result_dir"] = t.get("local_result_dir") or rds[0]
                candidates.append(candidate)
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
    monopolize the card before peak is observed. Once peak_vram_mb >= 100 or scheduler log
    progress proves the task has passed startup, nvidia-smi's aggregate card usage takes over."""
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
        if t.get("last_progress_line") or int(t.get("runtime_current_unit") or 0) > 0:
            # The startup reserve is only a launch-window guard. On WSL/local NVIDIA,
            # per-PID VRAM can stay invisible even while aggregate card memory is
            # accurate; keeping the synthetic reserve after tqdm/iter progress double
            # counts long-running jobs and blocks useful packing.
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
    """Resource-pressure rollback.

    If a GPU is over the 1/3 memory freeze line and multiple scheduler-owned
    tasks are sharing it, requeue newly-launched or no-progress colocated work.
    This is intentionally about memory only; util saturation alone is not enough
    to evict anything. Stable progress-bearing tasks are protected so a fresh
    dispatch cannot knock down a long-running experiment that had already been
    making progress on the GPU.

    Also applies the same idea at node RAM level: if probed free RAM is below
    configured headroom minus grace, requeue one scheduler-owned task on that
    node. RAM victim selection is RAM-pressure-first and checkpoint-aware; a
    small no-ETA task should not be easier to kill than a RAM-heavy sibling.
    """
    evicted = []
    evicted_set = set()

    def _gpu_evict_candidate(t):
        elapsed = _effective_elapsed_s(t)
        if (
            elapsed >= max(0, int(GPU_EVICT_STABLE_PROGRESS_MIN_AGE_S))
            and _task_has_progress_evidence(t)
        ):
            return False
        if elapsed <= max(0, int(GPU_EVICT_ROLLBACK_MAX_AGE_S)):
            return True
        return True

    def _gpu_victim_key(t):
        eta = int(t.get("eta_seconds") or 0)
        # Unknown ETA is treated as very long: we cannot prove it is close to done.
        eta_rank = eta if eta > 0 else 10 ** 12
        progress = _task_progress_ratio(t)
        progress_rank = -(progress if progress is not None else 0.0)
        no_progress_rank = 0 if _task_has_progress_evidence(t) else 1
        # Prefer rolling back the freshest colocated launch. This turns the
        # freeze-line path into a failed-placement rollback instead of a
        # preemption policy for long-running work.
        return (float(t.get("started_at") or 0), no_progress_rank, progress_rank, eta_rank)

    def _ram_victim_key(t):
        progress = _task_progress_ratio(t)
        progress_rank = -(progress if progress is not None else 0.0)
        eta = int(t.get("eta_seconds") or 0)
        eta_rank = eta if eta > 0 else 0
        return (_task_ram_pressure_mb(t), progress_rank, float(t.get("started_at") or 0), eta_rank)

    def _cpu_victim_key(t):
        progress = _task_progress_ratio(t)
        progress_rank = -(progress if progress is not None else 0.0)
        no_progress_rank = 0 if _task_has_progress_evidence(t) else 1
        return (
            no_progress_rank,
            progress_rank,
            float(t.get("started_at") or 0),
            max(0, int(t.get("cpu_cores") or DEFAULT_CPU_CORES)),
        )

    def _eligible_running(where):
        return [
            t for t in state["tasks"]
            if t.get("status") == "running"
            and t.get("started_at")
            and not t.get("auto_adopted")
            and not _is_slurm_managed(t)
            and t.get("id") not in evicted_set
            and where(t)
        ]

    def _evict(victim, reason, payload):
        _evict_to_queue(victim, state, reason, payload.get("kind") or "resource_pressure")
        victim["last_resource_eviction"] = payload
        evicted.append(victim["id"])
        evicted_set.add(victim["id"])

    for n in nodes:
        if not n.get("alive"):
            continue
        if n.get("name") == "local":
            budget = int(NODES.get("local", {}).get("cpu_cores") or n.get("total_cpu") or 0)
            if budget > 0:
                tasks_here = _eligible_running(lambda t: t.get("node") == "local")
                reserved = sum(
                    max(0, int(t.get("cpu_cores") or DEFAULT_CPU_CORES))
                    for t in tasks_here
                )
                if reserved > budget:
                    cpu_high, cpu_why = _local_cpu_pressure_high(n, LOCAL_GPU_HOST_CPU_EVICT_PCT)
                    if not cpu_high:
                        notify("local_cpu_over_reserved_no_evict", {
                            "reserved": reserved,
                            "budget": budget,
                            "reason": cpu_why,
                            "task_ids": [t.get("id") for t in tasks_here],
                        }, feishu_enabled=False)
                        continue
                while reserved > budget and len(tasks_here) > 1:
                    protected = [t for t in tasks_here if _task_evict_loss_protected(t)]
                    protected_ids = {id(t) for t in protected}
                    candidates = [t for t in tasks_here if id(t) not in protected_ids] or tasks_here
                    if not candidates:
                        break
                    victim = max(candidates, key=_cpu_victim_key)
                    cpu_payload = {
                        "kind": "local_cpu_budget",
                        "task_id": victim["id"],
                        "node": "local",
                        "cpu_reserved": reserved,
                        "cpu_budget": budget,
                        "same_node_tasks": [_summarize_task_for_resource_log(t) for t in tasks_here],
                        "protected_evict_loss_task_ids": [t.get("id") for t in protected],
                        "selection": "least_progress_then_newest_then_highest_cpu",
                    }
                    reason = (
                        f"evicted from local after CPU budget breach "
                        f"(reserved={reserved}/{budget}); "
                        f"victim selected by least progress / newest / highest CPU"
                    )
                    freed = max(0, int(victim.get("cpu_cores") or DEFAULT_CPU_CORES))
                    _evict(victim, reason, cpu_payload)
                    reserved = max(0, reserved - freed)
                    tasks_here = [t for t in tasks_here if t.get("id") != victim.get("id")]
                    if not tasks_here:
                        break
        for g in n["gpus"]:
            freeze = _gpu_freeze_line_mb(int(g.get("total_mb") or 0))
            occupied = g["used_mb"] > 100
            mem_over = occupied and freeze > 0 and g["used_mb"] >= freeze
            if not mem_over:
                continue
            tasks_here = _eligible_running(
                lambda t, node=n["name"], gpu=g["idx"]: t.get("node") == node and t.get("gpu_idx") == gpu
            )
            if len(tasks_here) < 2:
                continue  # single big task on the GPU — design exception, leave it
            candidates = [
                t for t in tasks_here
                if _gpu_evict_candidate(t)
                and not _task_ignores_one_third_pack_rule(t, NODES.get(t.get("node") or "", {}))
            ]
            if not candidates:
                continue
            protected = [t for t in tasks_here if t not in candidates]
            victim = max(candidates, key=_gpu_victim_key)
            gpu_payload = {
                "kind": "gpu_one_third",
                "task_id": victim["id"],
                "node": n["name"],
                "gpu_idx": g["idx"],
                "gpu": _gpu_threshold_snapshot(g),
                "same_gpu_tasks": [_summarize_task_for_resource_log(t) for t in tasks_here],
                "protected_stable_progress_task_ids": [t.get("id") for t in protected],
                "selection": "newest_launch_rollback_then_no_progress_then_least_progress",
            }
            reason = (
                f"evicted from {n['name']}:GPU{g['idx']} after GPU memory freeze-line breach "
                f"(mem={_format_mem_gb(g['used_mb'])}/{_format_mem_gb(g['total_mb'])}, "
                f"freeze={_format_mem_gb(freeze)}, util={g.get('util_pct','?')}%); "
                f"victim selected by newest launch rollback / no progress / least progress"
            )
            _evict(victim, reason, gpu_payload)

        ram = _node_ram_snapshot(n)
        if not ram or not ram.get("below_eviction_headroom"):
            continue
        tasks_here = _eligible_running(lambda t, node=n["name"]: t.get("node") == node)
        if len(tasks_here) < 2:
            continue
        protected = [t for t in tasks_here if _task_evict_loss_protected(t)]
        protected_ids = {id(t) for t in protected}
        candidates = [t for t in tasks_here if id(t) not in protected_ids] or tasks_here
        victim = max(candidates, key=_ram_victim_key)
        ram_payload = {
            "kind": "node_ram_headroom",
            "task_id": victim["id"],
            "node": n["name"],
            "node_ram": ram,
            "same_node_tasks": [_summarize_task_for_resource_log(t) for t in tasks_here],
            "protected_evict_loss_task_ids": [t.get("id") for t in protected],
            "protected_ckpt_task_ids": [t.get("id") for t in protected if _ckpt_task_evict_protected(t)],
            "selection": "highest_ram_pressure_then_least_progress_then_newest_loss_aware",
        }
        reason = (
            f"evicted from {n['name']} after RAM headroom breach "
            f"(free={_format_mem_gb(ram['free_mb'])} < "
            f"eviction_headroom={_format_mem_gb(ram['eviction_headroom_mb'])}, "
            f"headroom={_format_mem_gb(ram['headroom_mb'])}, "
            f"grace={_format_mem_gb(ram['headroom_grace_mb'])}); "
            f"victim selected by RAM pressure / least progress / newest with loss protection"
        )
        _evict(victim, reason, ram_payload)
    return evicted

EVICT_RELAUNCH_COOLDOWN_S = int(os.environ.get("SCHED_EVICT_RELAUNCH_COOLDOWN_S", str(30 * 60)))
LOCAL_CPU_EVICT_NODE_COOLDOWN_S = int(os.environ.get(
    "SCHED_LOCAL_CPU_EVICT_NODE_COOLDOWN_S", str(30 * 60)))
LOCAL_GPU_HOST_CPU_BLOCK_PCT = int(os.environ.get("SCHED_LOCAL_GPU_HOST_CPU_BLOCK_PCT", "85"))
LOCAL_GPU_HOST_CPU_EVICT_PCT = int(os.environ.get("SCHED_LOCAL_GPU_HOST_CPU_EVICT_PCT", "92"))
PREEMPT_QUEUE_WAIT_MIN = 5      # high-prio waits > N min before we consider preempting
PREEMPT_VICTIM_MIN_AGE_MIN = 10 # don't evict a task younger than this (let it settle)
PREEMPT_VICTIM_MAX_AGE_MIN = 240 # don't evict a task that's been stable too long (probably load-bearing)
PREEMPT_MAX_VICTIMS_PER_DISPATCH = 3  # cap chain-evictions; 1 round shouldn't kill an entire node

def _evict_to_queue(victim, state, reason, eviction_kind: str = "preempt"):
    """Send a running task back to queue WITHOUT incrementing retry_count or marking crash.
    Used by preemption — task didn't fail, we just made room for higher priority. Kills its
    PIDs, resets running fields, sets last_block_reason for visibility."""
    # Phase 3.2.1: release the cross-scheduler claim BEFORE clearing the
    # node — release() needs task["node"] still set to know where to ssh.
    try:
        _release_task_claims_and_intents(victim)
    except Exception:
        pass
    old_node = victim.get("node")
    old_pids = list(_task_pids(victim))
    actor = _record_task_kill_actor(victim, eviction_kind, reason)
    ok, kill_msg = _kill_task_processes(victim, timeout=15)
    notify("task_killed", {
        "task_id": victim.get("id"),
        "node": old_node,
        "pids": old_pids,
        "actor": actor,
        "action": eviction_kind,
        "reason": reason,
        "kill_ok": ok,
        "kill_msg": kill_msg,
    }, feishu_enabled=False)
    victim["status"] = "queued"
    for k in ("node", "gpu_idx", "process_group", "log_path", "started_at", "finished_at", "_diagnosis"):
        victim[k] = None
    victim["remote_pids"] = []
    victim["alive_pids"] = []
    victim["peak_vram_mb"] = 0
    victim["peak_ram_mb"] = 0
    _set_current_usage(victim, 0, 0, 0.0)
    _clear_live_eta_fields(victim, clear_runtime_projection=True)
    victim["notified_done"] = False
    victim["last_block_reason"] = reason
    now = time.time()
    victim["last_evicted_at"] = now
    victim["last_eviction_kind"] = eviction_kind
    if eviction_kind == "local_cpu_budget":
        victim.pop("evict_cooldown_until", None)
        node_cooldown = max(0, int(LOCAL_CPU_EVICT_NODE_COOLDOWN_S))
        if old_node and node_cooldown > 0:
            raw = victim.get("evict_node_cooldowns")
            if not isinstance(raw, dict):
                raw = {}
                victim["evict_node_cooldowns"] = raw
            raw[str(old_node)] = now + node_cooldown
        return
    cooldown = max(0, int(EVICT_RELAUNCH_COOLDOWN_S))
    if cooldown > 0:
        victim["evict_cooldown_until"] = now + cooldown
    else:
        victim.pop("evict_cooldown_until", None)


def _eviction_cooldown_block_reason(task: dict, now: Optional[float] = None) -> str:
    until = float(task.get("evict_cooldown_until") or 0)
    if until <= 0:
        return ""
    now = time.time() if now is None else now
    if until <= now:
        task.pop("evict_cooldown_until", None)
        return ""
    remain = int(max(0, until - now))
    return (
        f"recently {task.get('last_eviction_kind') or 'evicted'}; "
        f"cooling down {remain}s before relaunch to avoid kill/relaunch thrash"
    )


def _evict_node_cooldown_block_reason(task: dict, node: str, now: Optional[float] = None,
                                      node_state: Optional[dict] = None) -> str:
    """Block only the node that just evicted this task.

    Local CPU rollback should not globally cool the task down: remote GPU nodes
    may become available immediately. But without a local-only cooldown, the
    dispatcher can relaunch the task onto local and the post-dispatch budget
    guard will kill it again in a tight loop.
    """
    if not node:
        return ""
    now = time.time() if now is None else now
    raw = task.get("evict_node_cooldowns")
    if not isinstance(raw, dict):
        raw = {}
    until = 0.0
    try:
        until = float(raw.get(node) or 0)
    except Exception:
        until = 0.0
    if until <= 0 and node == "local" and task.get("last_eviction_kind") == "local_cpu_budget":
        try:
            last = float(task.get("last_evicted_at") or 0)
        except Exception:
            last = 0.0
        if last > 0:
            until = last + max(0, int(LOCAL_CPU_EVICT_NODE_COOLDOWN_S))
            if until > now:
                raw[node] = until
                task["evict_node_cooldowns"] = raw
    if (
        until > now
        and node == "local"
        and task.get("last_eviction_kind") == "local_cpu_budget"
        and node_state is not None
    ):
        high, _ = _local_cpu_pressure_high(node_state, LOCAL_GPU_HOST_CPU_BLOCK_PCT)
        if not high:
            raw.pop(node, None)
            if raw:
                task["evict_node_cooldowns"] = raw
            else:
                task.pop("evict_node_cooldowns", None)
            return ""
    if until <= 0:
        return ""
    if until <= now:
        if raw:
            raw.pop(node, None)
            if raw:
                task["evict_node_cooldowns"] = raw
            else:
                task.pop("evict_node_cooldowns", None)
        return ""
    remain = int(max(0, until - now))
    return (
        f"recently {task.get('last_eviction_kind') or 'evicted'} on {node}; "
        f"cooling down {remain}s before relaunching there"
    )

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
        node_state = next((n for n in nodes if n.get("name") == node_pin), None)
        node_info = NODES.get(node_pin, {})
        free_cpu = int((node_state or {}).get("free_cpu") or 0)
        free_ram = int((node_state or {}).get("free_ram_mb") or 0)
        headroom = _node_ram_headroom_mb(node_state or {}, node_info) if node_info else 0
        schedulable_ram = max(0, free_ram - headroom)
        ignore_cpu = _ignore_cpu_for_server_gpu_task(
            hi, node_state=node_state, node_info=node_info, node_name=node_pin,
            gpu_idx=hi.get("require_gpu_idx") if hi.get("require_gpu_idx") is not None else hi.get("gpu_idx"))
        cpu_deficit = 0 if ignore_cpu else max(0, int(cpu_need) - free_cpu)
        ram_deficit = max(0, int(ram_need) - schedulable_ram)
        if cpu_deficit <= 0 and ram_deficit <= 0:
            continue
        cpu_acc = ram_acc = 0
        wait_min = int((now - hi["submitted_at"]) / 60)
        for _ in range(PREEMPT_MAX_VICTIMS_PER_DISPATCH - len(evicted)):
            remaining_cpu = max(0, cpu_deficit - cpu_acc)
            remaining_ram = max(0, ram_deficit - ram_acc)
            if remaining_cpu <= 0 and remaining_ram <= 0:
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
            protected_ids = {id(t) for t in victims if _task_evict_loss_protected(t)}
            pool = [t for t in victims if id(t) not in protected_ids] or victims

            def _preempt_victim_key(t):
                cpu = int(t.get("cpu_cores") or DEFAULT_CPU_CORES)
                ram = _task_ram_pressure_mb(t)
                progress = _task_progress_ratio(t)
                progress_rank = -(progress if progress is not None else 0.0)
                started = float(t.get("started_at") or 0)
                if remaining_ram > 0:
                    return (
                        min(ram, remaining_ram),
                        min(cpu, remaining_cpu) if remaining_cpu > 0 else 0,
                        ram,
                        progress_rank,
                        started,
                    )
                return (
                    min(cpu, remaining_cpu),
                    progress_rank,
                    -ram,
                    started,
                )

            victim = max(pool, key=_preempt_victim_key)
            cpu_freed = victim.get("cpu_cores") or DEFAULT_CPU_CORES
            ram_freed = _task_ram_pressure_mb(victim) or DEFAULT_RAM_MB
            _evict_to_queue(victim, state,
                            f"preempted by {hi['id']} (high-prio waited {wait_min}min; "
                            f"deficit cpu={cpu_deficit}, ram={_format_mem_gb(ram_deficit)})",
                            "preempt_high_priority")
            evicted.append({"id": victim["id"], "node": node_pin,
                            "cpu_freed": cpu_freed, "ram_freed": ram_freed,
                            "target_id": hi.get("id"),
                            "cpu_deficit": cpu_deficit,
                            "ram_deficit": ram_deficit,
                            "protected_skipped": [t.get("id") for t in victims if id(t) in protected_ids]})
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
    running_per_gpu = _Counter(
        (t.get("node"), t.get("gpu_idx")) for t in state["tasks"]
        if _counts_against_node_concurrency(t) and t.get("gpu_idx") is not None
    )
    cpu_labor_names = _cpu_labor_node_names()
    cpu_slot_names = set(cpu_labor_names)
    if "local" in NODES:
        cpu_slot_names.add("local")
    cpu_slot_reserved = _Counter()
    for _t in state["tasks"]:
        if _t.get("status") not in ("running", "launching"):
            continue
        _node = _t.get("node") or _t.get("assigned_node")
        if _node not in cpu_slot_names:
            continue
        cpu_slot_reserved[_node] += max(0, int(_t.get("cpu_cores") or DEFAULT_CPU_CORES))
    # Phase 2.16/3.4.13: count OUR slurm-pending tasks per node and split by
    # CPU/GPU bucket. pick_placement throttles further dispatch only when the
    # matching bucket is full, so CPU-only work can proceed behind pending GPU jobs.
    slurm_pending_per_node = _count_slurm_pending_per_node(state)
    for n in nodes:
        n["running_count"] = running_per_node.get(n["name"], 0)
        for g in n.get("gpus") or []:
            g["running_task_count"] = running_per_gpu.get((n["name"], g.get("idx")), 0)
        if n["name"] in cpu_slot_names:
            total_cpu = int(NODES.get(n["name"], {}).get("cpu_cores") or n.get("total_cpu") or 0)
            if total_cpu > 0:
                raw_reserved_cpu = int(cpu_slot_reserved.get(n["name"], 0))
                reserved_cpu = min(total_cpu, raw_reserved_cpu)
                n["observed_free_cpu"] = n.get("free_cpu")
                n["observed_loadavg"] = n.get("loadavg")
                n["cpu_slot_accounting"] = True
                n["cpu_slot_reserved"] = raw_reserved_cpu
                n["total_cpu"] = total_cpu
                if n["name"] != "local":
                    n["free_cpu"] = max(0, total_cpu - reserved_cpu)
                    n["loadavg"] = float(reserved_cpu)
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
                        "cpu_freed": ev["cpu_freed"], "ram_freed": ev["ram_freed"],
                        "target_id": ev.get("target_id"),
                        "cpu_deficit": ev.get("cpu_deficit"),
                        "ram_deficit": ev.get("ram_deficit"),
                        "protected_skipped": ev.get("protected_skipped") or []})
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
            if not t.get("est_vram_mb_explicit"):
                new_est = int(h["vram_mb"])
                if new_est != t.get("est_vram_mb"):
                    t["est_vram_mb"] = new_est
        else:
            # NO own history yet: cascade is a guess from siblings/project. Only allow it to
            # LOWER the stored est, never raise it. Reason: if a guess says 4096 but user / prior
            # state said 512, raising hurts schedulability and the eviction mechanism would
            # immediately catch a real OOM. Symmetric trust would let one giant sibling pollute
            # all small siblings' est upward forever.
            if not t.get("est_vram_mb_explicit"):
                new_est = _effective_est_vram(t, state, history_cache)
                cur = t.get("est_vram_mb") or 0
                if new_est and 0 < new_est < cur:
                    t["est_vram_mb"] = new_est
        if t.get("ram_mb_explicit"):
            pass
        elif isinstance(h, dict) and h.get("ram_mb"):
            new_ram = int(h["ram_mb"])
            if new_ram != t.get("ram_mb"):
                t["ram_mb"] = new_ram
        else:
            # Live RAM is asymmetric: under-estimating it can trigger node-level rollback
            # and immediately make the next dispatch worse. If currently-running siblings
            # prove this queued task's stale estimate is too low, raise the floor before
            # placement. Historical/project guesses still only lower below.
            live_ram = _live_sibling_ram_floor(t, state)
            cur_ram = int(t.get("ram_mb") or 0)
            if live_ram and live_ram > cur_ram * 1.1:
                t["ram_mb"] = live_ram
                t["last_resource_estimate_update"] = {
                    "ts": time.time(),
                    "kind": "ram_live_sibling_floor",
                    "old_ram_mb": cur_ram,
                    "new_ram_mb": live_ram,
                }
                continue
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
    resume_scan_cache = {}
    for t in queued:
        if _clear_disallowed_cpu_fallback_selection(t):
            t["last_block_reason"] = (
                "cleared stale CPU fallback placement; task has vram>0 and "
                "will wait for GPU/local placement"
            )
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
        terminal_artifact_reason = _same_run_identity_live_artifact_reason(t, state, run_key)
        if terminal_artifact_reason:
            t["last_block_reason"] = terminal_artifact_reason
            events.append({"type": "blocked", "task_id": t["id"], "task": t,
                           "reason": terminal_artifact_reason})
            continue
        cooldown_block = _eviction_cooldown_block_reason(t)
        if cooldown_block:
            t["last_block_reason"] = cooldown_block
            events.append({"type": "blocked", "task_id": t["id"], "task": t, "reason": cooldown_block})
            continue
        wait_file_block = _queued_wait_for_file_block_reason(t)
        if wait_file_block:
            t["last_block_reason"] = wait_file_block
            events.append({"type": "blocked", "task_id": t["id"], "task": t, "reason": wait_file_block})
            continue
        cpu_training_block = _queued_cpu_training_block_reason(t)
        if cpu_training_block:
            t["last_block_reason"] = cpu_training_block
            events.append({"type": "blocked", "task_id": t["id"], "task": t, "reason": cpu_training_block})
            continue
        resume_locations = []
        resume_errors = {}
        resume_nodes = set()
        placement = None
        if _task_requires_resume_scan(t):
            # Cheap capacity gate before slow checkpoint scans. If no node/GPU
            # can accept this task right now, scanning every remote filesystem is
            # pure latency and can starve the few tasks that could launch.
            pre_scan_placement = pick_placement(t, nodes)
            if pre_scan_placement is not None:
                resume_locations, resume_errors = _refresh_resume_locations_for_task(
                    t, nodes, resume_scan_cache)
                if resume_errors and not resume_locations:
                    reason = (
                        "resume checkpoint scan failed on all checked nodes; refusing to dispatch "
                        "because a checkpoint may exist on an unchecked server: "
                        + "; ".join(f"{n}: {msg}" for n, msg in sorted(resume_errors.items())[:3])
                    )
                    t["last_block_reason"] = reason
                    events.append({"type": "blocked", "task_id": t["id"], "task": t, "reason": reason})
                    continue
                resume_nodes = {loc.get("node") for loc in (resume_locations or []) if loc.get("node")}
                if resume_nodes and t.get("require_node") and t.get("require_node") not in resume_nodes:
                    require_node = t.get("require_node")
                    stage_state, source_loc, stage_msg = _resume_checkpoint_stage_check(
                        t, require_node, resume_locations)
                    if stage_state == "ready" and source_loc:
                        _record_staged_resume_location(t, require_node, source_loc)
                        resume_locations = list(t.get("resume_locations") or [])
                        resume_nodes = {loc.get("node") for loc in resume_locations if loc.get("node")}
                    else:
                        if stage_state == "needs_stage":
                            reason = (
                                f"resume checkpoint exists on {','.join(sorted(resume_nodes))}, "
                                f"but require_node={require_node} has no matching checkpoint yet; "
                                "awaiting small-checkpoint staging before remote launch"
                            )
                        elif stage_state == "cap_exceeded":
                            reason = (
                                f"resume checkpoint exists on {','.join(sorted(resume_nodes))}, "
                                f"but require_node={require_node} has no checkpoint and ckpt_dir "
                                f"is too large for automatic staging; keep on checkpoint-local node"
                            )
                        elif stage_state == "stage_failed":
                            reason = (
                                f"resume checkpoint staging to require_node={require_node} failed: "
                                f"{stage_msg[:220]}"
                            )
                        else:
                            reason = (
                                f"resume checkpoint exists on {','.join(sorted(resume_nodes))}, "
                                f"but require_node={require_node} has no matching checkpoint; "
                                "refusing to launch there because it would likely restart from step 0"
                            )
                        t["last_block_reason"] = reason
                        events.append({"type": "blocked", "task_id": t["id"], "task": t, "reason": reason})
                        continue
                placement = pick_placement(t, nodes)
        else:
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
                node_evict_cooldown = _evict_node_cooldown_block_reason(t, n["name"], node_state=n)
                if node_evict_cooldown:
                    reasons.append(f"{n['name']}={node_evict_cooldown}"); continue
                if n["name"] in (blocked or set()):
                    reasons.append(f"{n['name']}=blocklisted"); continue
                # Phase 2.3 P1: slurm nodes don't go through local capacity gate, so showing
                # "GPU0=1/3 mem locked" is misleading (we never probed slurm-side capacity).
                # If a slurm node is alive + not blocklisted but pick_placement still returned
                # None, the only legitimate reason is require_node mismatch — surface that.
                if not _requires_local_capacity_check(n["name"], t, n):
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
                cpu_fallback_node = (
                    (t.get("est_vram_mb") or 0) > 0
                    and (_node_is_windows(n["name"]) or not n.get("gpus") or NODES[n["name"]].get("max_vram_per_task") == 0)
                )
                if cpu_fallback_node:
                    cpu_fb_block = _node_cpu_fallback_block_reason(t, n["name"], NODES[n["name"]])
                    if cpu_fb_block:
                        reasons.append(f"{n['name']}={cpu_fb_block}")
                        continue
                    ok_node, why_node = _node_resources_ok(t, n, NODES[n["name"]])
                    if not ok_node:
                        reasons.append(f"{n['name']}={why_node}")
                    else:
                        reasons.append(f"{n['name']}=cpu-fallback fits?(unexpected)")
                    continue
                gpu_policy_block = _node_gpu_task_block_reason(t, n["name"], NODES[n["name"]])
                if gpu_policy_block:
                    reasons.append(f"{n['name']}={gpu_policy_block}")
                    continue
                ok_node, why_node = _node_resources_ok(t, n, NODES[n["name"]])
                if not ok_node:
                    reasons.append(f"{n['name']}={why_node}"); continue
                if (t.get("est_vram_mb") or 0) > 0:
                    gpu_reasons = []
                    for g in n["gpus"]:
                        sub = []
                        algo_reason = _algorithm_gpu_fit_block_reason(t, g, NODES[n["name"]])
                        if algo_reason:
                            sub.append(algo_reason)
                        max_tasks_per_gpu = NODES[n["name"]].get("max_tasks_per_gpu")
                        if max_tasks_per_gpu is not None:
                            try:
                                gpu_task_count = int(g.get("running_task_count") or 0)
                                gpu_task_cap = int(max_tasks_per_gpu)
                            except Exception:
                                gpu_task_count = 0
                                gpu_task_cap = 0
                            if gpu_task_cap > 0 and gpu_task_count >= gpu_task_cap:
                                sub.append(f"tasks {gpu_task_count}/{gpu_task_cap}")
                        if not _gpu_fits(t, g, NODES[n["name"]]):
                            freeze = _gpu_freeze_line_mb(int(g.get("total_mb") or 0))
                            if g["used_mb"] > 100 and (
                                    g["used_mb"] >= freeze
                                    or g["used_mb"] + (t.get("est_vram_mb") or 0) >= freeze):
                                sub.append(
                                    f"1/3+grace mem ({g['used_mb']}+{t.get('est_vram_mb') or 0}>={freeze}MB)")
                            util_limit = _node_gpu_util_limit(NODES[n["name"]])
                            if util_limit is not None and g["used_mb"] > 100 and g.get("util_pct", 0) >= util_limit:
                                sub.append(f"util {g['util_pct']}%")
                            if g["free_mb"] < (t.get("est_vram_mb") or 0) + VRAM_MARGIN_MB:
                                sub.append(f"free<est+margin")
                            cap = NODES[n["name"]].get("max_vram_per_task")
                            if cap is not None and t.get("est_vram_mb", 0) > cap:
                                sub.append(f"per-task cap {cap}")
                        if sub:
                            gpu_reasons.append(f"GPU{g['idx']}=" + "&".join(sub))
                    if gpu_reasons:
                        reasons.append(f"{n['name']}=node-ok-but-" + "/".join(gpu_reasons))
                    else:
                        reasons.append(f"{n['name']}=fits?(unexpected)")
            pin = f"require={require} " if require else (f"prefer={prefer} " if prefer else "")
            t["last_block_reason"] = f"no fit ({pin}prio={t.get('priority','normal')}): " + " | ".join(reasons[:4])
            events.append({"type": "no_fit", "task_id": t["id"], "task": t})
            continue
        t["node"], t["gpu_idx"] = placement
        t["placement_algorithm"] = _algorithm_name()
        t["placement_algorithm_config"] = _algorithm_config_snapshot()
        selected_info = NODES.get(t.get("node"), {})
        selected_cpu_fallback = (
            int(t.get("est_vram_mb") or 0) > 0
            and t.get("gpu_idx") is None
            and not _node_cpu_fallback_block_reason(t, t.get("node"), selected_info)
        )
        if selected_cpu_fallback:
            t["cpu_fallback_selected"] = True
            t["cpu_fallback_original_vram_mb"] = int(t.get("est_vram_mb") or 0)
            t["cpu_fallback_capability"] = _task_cpu_fallback_capability(t)
        else:
            t.pop("cpu_fallback_selected", None)
            t.pop("cpu_fallback_original_vram_mb", None)
            t.pop("cpu_fallback_capability", None)
        picked_state = next((n for n in nodes if n.get("name") == t.get("node")), None)
        selected_gpu = None
        if picked_state is not None and t.get("gpu_idx") is not None:
            for g in picked_state.get("gpus") or []:
                try:
                    if int(g.get("idx")) == int(t.get("gpu_idx")):
                        selected_gpu = g
                        break
                except Exception:
                    continue
        if selected_gpu is not None:
            algo_audit = _algorithm_selected_gpu_audit(t, picked_state, selected_gpu)
            if algo_audit:
                t["placement_algorithm_audit"] = algo_audit
            else:
                t.pop("placement_algorithm_audit", None)
        else:
            t.pop("placement_algorithm_audit", None)
        _assign_windows_pin_plan(t, state, picked_state)
        if resume_nodes and t.get("node") not in resume_nodes:
            selected_node = t.get("node")
            stage_state, source_loc, stage_msg = _resume_checkpoint_stage_check(
                t, selected_node, resume_locations)
            if stage_state == "ready" and source_loc:
                _record_staged_resume_location(t, selected_node, source_loc)
                resume_locations = list(t.get("resume_locations") or [])
                resume_nodes = {loc.get("node") for loc in resume_locations if loc.get("node")}
            else:
                if stage_state == "needs_stage":
                    reason = (
                        f"resume checkpoint exists on {','.join(sorted(resume_nodes))}, "
                        f"but selected node {selected_node} has no matching checkpoint yet; "
                        "awaiting small-checkpoint staging before remote launch"
                    )
                elif stage_state == "cap_exceeded":
                    reason = (
                        f"resume checkpoint exists on {','.join(sorted(resume_nodes))}, "
                        f"but selected node {selected_node} lacks it and ckpt_dir is too "
                        "large for automatic staging; waiting for checkpoint-local capacity"
                    )
                elif stage_state == "stage_failed":
                    reason = (
                        f"resume checkpoint staging to selected node {selected_node} failed: "
                        f"{stage_msg[:220]}"
                    )
                else:
                    reason = (
                        f"resume checkpoint exists on {','.join(sorted(resume_nodes))}, "
                        f"but selected node {selected_node} has no matching checkpoint; "
                        "waiting for a checkpoint-local node instead of launching a fresh run"
                    )
                t["last_block_reason"] = reason
                _release_task_claims_and_intents(t)
                t["node"] = None
                t["gpu_idx"] = None
                events.append({"type": "blocked", "task_id": t["id"], "task": t, "reason": reason})
                continue
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
        loc = _resume_location_for_node(t, t.get("node"))
        resume = None if t.get("skip_resume_scan") else ((loc or {}).get("path") or find_resume(t))
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
                    f"launch staging: cwd not yet staged to {target}; "
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
                    "reason": f"awaiting outside-lock staging to {target}",
                })
                continue
            # stage_state == "ready" — fall through to launch
        pre_launch_snapshot = _resource_snapshot_for_placement(
            state, nodes, t, t.get("node"), t.get("gpu_idx"), "pre_launch",
        )
        t["last_launch_pre_snapshot"] = pre_launch_snapshot
        events.append({
            "type": "pre_launch_snapshot",
            "task_id": t["id"],
            "snapshot": pre_launch_snapshot,
        })
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
        t.pop("evict_cooldown_until", None)
        t.pop("evict_node_cooldowns", None)
        if not str(t.get("last_block_reason") or "").startswith("warn:"):
            t.pop("last_block_reason", None)
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
            cpu_debit = 0 if (n["name"] == "local" and _task_is_gpu_capacity_task(t)) else t.get("cpu_cores", DEFAULT_CPU_CORES)
            n["free_cpu"] = max(0, n.get("free_cpu", 0) - cpu_debit)
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
                g["running_task_count"] = int(g.get("running_task_count") or 0) + 1
        post_launch_snapshot = _resource_snapshot_for_placement(
            state, nodes, t, t.get("node"), t.get("gpu_idx"), "post_launch_estimated",
        )
        pre_gpu = (pre_launch_snapshot or {}).get("gpu") or {}
        post_gpu = (post_launch_snapshot or {}).get("gpu") or {}
        pre_ram = (pre_launch_snapshot or {}).get("node_ram") or {}
        post_ram = (post_launch_snapshot or {}).get("node_ram") or {}
        post_launch_snapshot["crossed_one_third_from_pre"] = (
            bool(post_gpu.get("over_one_third")) and not bool(pre_gpu.get("over_one_third"))
        )
        post_launch_snapshot["crossed_ram_headroom_from_pre"] = (
            bool(post_ram.get("below_headroom")) and not bool(pre_ram.get("below_headroom"))
        )
        t["last_launch_post_snapshot"] = post_launch_snapshot
        t["last_resource_snapshot"] = post_launch_snapshot
        events.append({
            "type": "post_launch_snapshot",
            "task_id": t["id"],
            "snapshot": post_launch_snapshot,
        })
    return events, len(queued)

def _print_node_summary(nodes):
    print("=== nodes ===")
    for n in nodes:
        if not n["alive"]:
            print(f"  {n['name']:11s} DOWN ({n.get('error','?')})"); continue
        if n.get("slurm_cluster"):
            print(f"  {n['name']:11s} {_format_slurm_cluster_summary(n)}")
            continue
        gpu_parts = []
        for g in n["gpus"]:
            mem_pct = int(round(100 * g["used_mb"] / max(g["total_mb"], 1)))
            gpu_parts.append(
                f"GPU{g['idx']}={_format_mem_gb(g['used_mb'])}/{_format_mem_gb(g['total_mb'])}"
                f"(free={_format_mem_gb(g.get('free_mb', 0))}, mem:{mem_pct}%, util:{g['util_pct']}%)")
        gpu_str = ", ".join(gpu_parts) if gpu_parts else "CPU-only"
        load = n.get("loadavg", 0)
        reserve = int(NODES.get(n.get("name"), {}).get("reserved_cpu_cores") or 0)
        cpu_parts = []
        if isinstance(load, (int, float)):
            cpu_parts.append(f"load {load:.1f}")
        host_cpu = n.get("host_cpu_load_pct")
        if host_cpu is not None:
            wsl_load = n.get("wsl_loadavg")
            if isinstance(wsl_load, (int, float)):
                cpu_parts[-1:] = [f"wsl_load {wsl_load:.1f}"]
            cpu_parts.append(f"host {int(host_cpu)}%")
        if n.get("probe_fallback"):
            cpu_parts.append(str(n.get("probe_fallback")))
        if reserve:
            cpu_parts.append(f"reserve {reserve}")
        cpu_tail = f"({', '.join(cpu_parts)})" if cpu_parts else ""
        cpu_str = f"cpu={n.get('free_cpu', '?')}/{n.get('total_cpu', '?')}{cpu_tail}"
        ram_str = _format_node_ram_summary(n)
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
    owner = c.get("owner") or "?"
    return (f"{owner}/{c.get('scheduler_id','?')}/{c.get('task_id','?')}:{loc} "
            f"vram={_format_mem_gb(c.get('vram_mb') or 0)} "
            f"cpu={int(c.get('cpu_cores') or 0)} "
            f"ram={_format_mem_gb(c.get('ram_mb') or 0)} "
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
    other = []
    try:
        own_sid = _ClaimManager.scheduler_id()
        other = [c for c in (active + pending + intents)
                 if c.get("scheduler_id") and c.get("scheduler_id") != own_sid]
    except Exception:
        other = []
    if other:
        owners = ",".join(sorted({str(c.get("owner") or "?") for c in other})[:3])
        parts.append(f"other_schedulers={len(other)} owner={owners}")
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
    if _node_is_windows(node):
        return False
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
            try:
                _record_result_artifacts(task)
            except Exception:
                pass
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
    if _node_is_windows(node):
        return False
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
    if _node_is_windows(node):
        return False
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
    try:
        _record_result_artifacts(task)
    except Exception:
        pass
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
        _clear_live_eta_fields(t, clear_runtime_projection=True)
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
                if _node_is_windows(n):
                    continue
                needed_docker.add((n, chosen))
        elif kind == "conda":
            if not spec_payload: continue
            # Skip non-existent local source — caller's mistake, eventual launch failure is
            # diagnosed through the normal ENV_MISSING path.
            if not Path(spec_payload).is_absolute(): continue
            if not Path(spec_payload).is_dir(): continue
            for n in candidates:
                if _node_is_windows(n):
                    continue
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
    _configure_algorithm(getattr(args, "algorithm", None) or None)
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
        _seed_pending_eta_from_history(state)
        nodes = probe_all()
        _remember_running_resource_snapshots(state, nodes)
        _reconcile_aggregate_only_vram(state, nodes)
        _reserve_inflight_vram(state, nodes)
        _print_node_summary(nodes)
        events, qcount = _do_dispatch(state, nodes)
        save_state(state)
    if recovered_launching:
        notify("launching_state_recovered", {"reverted_count": recovered_launching},
               feishu_enabled=False)
    for ev in events:
        if ev.get("type") in ("pre_launch_snapshot", "post_launch_snapshot"):
            notify(ev["type"], ev.get("snapshot") or {}, feishu_enabled=False)
    notify("dispatch_cycle", _dispatch_cycle_log_payload(state, nodes, events, qcount),
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
            reason = str(t.get("last_block_reason") or "no fit")
            if len(reason) > 240:
                reason = reason[:237] + "..."
            print(f"  [{tid}] WAIT  {reason} ({t['description'][:50]})")
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
                  f"(freed {ev.get('cpu_freed', 0)}c / {_format_mem_gb(ev.get('ram_freed', 0))})")
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
                f"(freed {payload.get('cpu_freed', 0)}c/{_format_mem_gb(payload.get('ram_freed', 0))})")
    if event_type == "task_killed":
        a = payload.get("actor") or {}
        return (f"[scheduler] kill {payload.get('task_id','?')} by "
                f"{a.get('label') or '?'} action={payload.get('action','?')} "
                f"node={payload.get('node','?')} reason={str(payload.get('reason',''))[:120]}")
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

_TASK_EVENT_PAYLOAD_KEYS = (
    "id", "status", "description", "project", "signature", "cmd", "cwd",
    "origin", "submitted_by", "submitted_host", "scheduler_id",
    "process_owner", "shared_account_suspect",
    "cpu_parallel_items", "cpu_parallel_total_items", "cpu_parallel_start",
    "cpu_parallel_logical_items", "cpu_parallel_item_multiplier",
    "cpu_parallel_end", "cpu_auto_workers", "cpu_parallel_waves",
    "cpu_parallel_physical_cores", "cpu_parallel_shard_index",
    "cpu_parallel_num_shards", "cpu_parallel_total_physical_cores",
    "cpu_parallel_last_wave_items", "cpu_batch_plan",
    "est_vram_mb", "ram_mb", "cpu_cores", "priority", "preferred_node",
    "require_node", "require_gpu_idx", "allow_gpu_over_one_third",
    "allowed_nodes", "stage_excludes", "reroute_on_node_down",
    "node_down_requeue_s", "node", "gpu_idx", "remote_pids", "slurm_job_id",
    "slurm_state", "log_path", "submitted_at", "started_at", "finished_at",
    "peak_vram_mb", "peak_ram_mb", "current_vram_mb", "current_ram_mb",
    "current_pcpu", "resume_from", "eta_seconds", "eta_source",
    "eta_confidence", "runtime_current_unit", "runtime_total_units",
    "placement_algorithm", "placement_algorithm_config", "placement_algorithm_audit",
    "last_progress_line", "result_artifacts", "result_artifacts_discovered_at",
    "parent_id", "retry_count", "requeued_as", "launch_fail_count",
    "failure_category", "last_block_reason", "notified_done", "notified_launch",
    "cancelled_by", "cancel_reason", "last_killed_by", "last_kill_action",
    "last_kill_reason",
)


def _compact_task_event_payload(task: dict) -> dict:
    """Small task payload for watcher.log task events.

    Full task records now carry resource snapshots and crash forensics. Those
    belong in queue/archive state and dedicated snapshot/forensics events; if
    task_done/task_launched writes them again, watcher.log becomes hard to read
    and rotates away useful history too quickly.
    """
    out = {k: task.get(k) for k in _TASK_EVENT_PAYLOAD_KEYS if k in task}
    diag = task.get("_diagnosis")
    if isinstance(diag, dict):
        out["_diagnosis"] = {
            k: diag.get(k)
            for k in ("is_crash", "reason", "lifetime_s", "log_size", "log_path", "success_marker")
            if k in diag
        }
    return out


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
    if _node_is_windows(name):
        return []
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
    if _node_is_windows(name):
        return {}
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
    if _node_is_windows(name):
        return []
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
        owners = sorted({p.get("owner") or me for p in procs})
        proc_owner = owners[0] if len(owners) == 1 else ",".join(owners[:3])
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
            "origin": "external",
            "submitted_by": proc_owner,
            "process_owner": proc_owner,
            "submitted_host": node,
            "scheduler_id": None,
            "shared_account_suspect": True,
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

def _cpu_ownership_snapshot(state, nodes) -> list:
    """Estimate per-node CPU attribution for JSONL logs.

    free_cpu/total_cpu come from live probes. scheduleurm/external tracked
    reservations come from queue state. The residual is "other/untracked" CPU:
    same account jobs from another shell, another user on a shared account, OS
    load, or probe noise. This is intentionally an estimate, but it is enough to
    explain "why did/didn't dispatch" after the fact.
    """
    own_sid = _ClaimManager.scheduler_id()
    tracked = {}
    for t in state.get("tasks", []):
        if t.get("status") not in ("running", "launching"):
            continue
        node = t.get("node") or t.get("assigned_node")
        if not node:
            continue
        cpu = max(0, int(t.get("cpu_cores") or DEFAULT_CPU_CORES))
        sid = t.get("scheduler_id")
        ours = (
            t.get("origin") == "scheduleurm"
            and not t.get("auto_adopted")
            and (not sid or sid == own_sid)
        )
        bucket = tracked.setdefault(node, {
            "ours_cpu": 0,
            "ours_tasks": [],
            "external_cpu": 0,
            "external_tasks": [],
        })
        rec = {
            "id": t.get("id"),
            "cpu_cores": cpu,
            "status": t.get("status"),
            "owner": t.get("process_owner") or t.get("submitted_by"),
            "scheduler_id": sid,
            "description": (t.get("description") or "")[:80],
        }
        if ours:
            bucket["ours_cpu"] += cpu
            bucket["ours_tasks"].append(rec)
        else:
            bucket["external_cpu"] += cpu
            bucket["external_tasks"].append(rec)

    out = []
    for n in nodes or []:
        name = n.get("name")
        bucket = tracked.get(name, {})
        try:
            total = int(n.get("total_cpu") or _node_physical_cores(name, n) or 0)
        except Exception:
            total = 0
        try:
            free = max(0, int(n.get("free_cpu") or 0))
        except Exception:
            free = 0
        if n.get("alive") is False:
            free = 0
        used_est = max(0, total - free)
        ours_cpu = int(bucket.get("ours_cpu") or 0)
        external_cpu = int(bucket.get("external_cpu") or 0)
        untracked_other = max(0, used_est - ours_cpu - external_cpu)
        over_reserved = max(0, ours_cpu + external_cpu - used_est)
        out.append({
            "node": name,
            "alive": n.get("alive"),
            "total_cpu": total,
            "logical_cpu": n.get("logical_cpu") or n.get("logical_cores"),
            "free_cpu": free,
            "used_cpu_est": used_est,
            "scheduleurm_cpu_reserved": ours_cpu,
            "scheduleurm_task_count": len(bucket.get("ours_tasks") or []),
            "scheduleurm_task_ids": [x.get("id") for x in (bucket.get("ours_tasks") or [])[:50]],
            "external_tracked_cpu_reserved": external_cpu,
            "external_tracked_task_count": len(bucket.get("external_tasks") or []),
            "external_tracked_task_ids": [x.get("id") for x in (bucket.get("external_tasks") or [])[:50]],
            "untracked_or_other_user_cpu_est": untracked_other,
            "over_reserved_cpu_est": over_reserved,
            "host_cpu_load_pct": n.get("host_cpu_load_pct"),
            "host_cpu_used_cores": n.get("host_cpu_used_cores"),
            "wsl_loadavg": n.get("wsl_loadavg"),
            "observed_free_cpu": n.get("observed_free_cpu"),
            "cpu_slot_reserved": n.get("cpu_slot_reserved"),
            "free_ram_mb": n.get("free_ram_mb"),
            "total_ram_mb": n.get("total_ram_mb"),
            "running_count": n.get("running_count"),
        })
    return out


def _resource_accounting_payload(state, nodes) -> dict:
    return {
        "running": sum(1 for t in state.get("tasks", []) if t.get("status") == "running"),
        "launching": sum(1 for t in state.get("tasks", []) if t.get("status") == "launching"),
        "queued": sum(1 for t in state.get("tasks", []) if t.get("status") == "queued"),
        "nodes": _cpu_ownership_snapshot(state, nodes),
    }


def _dispatch_cycle_log_payload(state, nodes, events, queued_count: int) -> dict:
    event_counts = {}
    launched = []
    blocked = []
    no_fit = []
    for ev in events or []:
        typ = ev.get("type")
        event_counts[typ] = event_counts.get(typ, 0) + 1
        if typ == "launched":
            t = ev.get("task") or {}
            launched.append({
                "task_id": ev.get("task_id") or t.get("id"),
                "node": t.get("node"),
                "gpu_idx": t.get("gpu_idx"),
                "cpu_cores": t.get("cpu_cores"),
                "ram_mb": t.get("ram_mb"),
                "est_vram_mb": t.get("est_vram_mb"),
                "placement_algorithm": t.get("placement_algorithm"),
                "placement_algorithm_audit": t.get("placement_algorithm_audit"),
                "description": (t.get("description") or "")[:100],
            })
        elif typ == "blocked":
            blocked.append({"task_id": ev.get("task_id"), "reason": ev.get("reason")})
        elif typ == "no_fit":
            t = ev.get("task") or {}
            no_fit.append({
                "task_id": ev.get("task_id") or t.get("id"),
                "reason": t.get("last_block_reason") or "no fit",
                "cpu_cores": t.get("cpu_cores"),
                "ram_mb": t.get("ram_mb"),
                "est_vram_mb": t.get("est_vram_mb"),
            })
    payload = _resource_accounting_payload(state, nodes)
    payload.update({
        "placement_algorithm": _algorithm_name(),
        "placement_algorithm_config": _algorithm_config_snapshot(),
        "queued_seen_by_dispatch": int(queued_count or 0),
        "event_counts": event_counts,
        "launched": launched[:100],
        "blocked": blocked[:100],
        "no_fit": no_fit[:100],
    })
    return payload


def _build_heartbeat_payload(state, nodes):
    running = sum(1 for t in state["tasks"] if t["status"] == "running")
    launching = sum(1 for t in state["tasks"] if t["status"] == "launching")
    queued = sum(1 for t in state["tasks"] if t["status"] == "queued")
    node_strs = []
    for n in nodes:
        if not n["alive"]:
            node_strs.append(f"{n['name']}:DOWN"); continue
        if n.get("slurm_cluster"):
            node_strs.append(f"{n['name']}:slurm gpu={n.get('free_gpus', 0)}/{n.get('total_gpus', 0)}")
            continue
        gpu_brief = "/".join(_format_mem_gb(g.get("used_mb", 0)) for g in n["gpus"])
        node_strs.append(f"{n['name']}:{gpu_brief}")
    return {
        "running": running,
        "launching": launching,
        "queued": queued,
        "nodes": node_strs,
        "cpu_accounting": _cpu_ownership_snapshot(state, nodes),
    }

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
        raw_cmd = t.get("cmd", "") or ""
        # Two cmd shapes to handle:
        #   (a) absolute path: /home/user/conda/envs/X/bin/python ... → probe that path directly
        #   (b) `conda run -n <env> python ...` → probe via `conda run -n <env> which python`
        probes: list[tuple[str, str]] = []  # (probe_cmd, label)
        target = t.get("node") or t.get("require_node") or t.get("preferred_node")
        if not target:
            continue
        if _node_is_windows(target):
            # WindowsBackend rewrites python/path tokens and validates cwd at
            # launch. This Linux bash smoke test would be a false failure.
            continue
        cmd = _apply_node_cmd_rewrites(target, raw_cmd)
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
    _configure_algorithm(getattr(args, "algorithm", None) or None)
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
          "last_resource_log_ts": 0, "interval": args.interval,
          "heartbeat": args.heartbeat,
          "resource_log_interval": getattr(args, "resource_log_interval", RESOURCE_LOG_INTERVAL_S)}
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
    with state_lock():
        state = load_state()
        recovered_launching_count += recover_stale_launching_tasks(state)
        # 1. Detect transitions running → done. update_running_tasks marks status; we record which IDs
        #    transitioned so we only notify the freshly-done ones.
        pre_status = {t["id"]: t["status"] for t in state["tasks"]}
        update_running_tasks(state)
        _seed_pending_eta_from_history(state)
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
        _remember_running_resource_snapshots(state, nodes)
        _reconcile_aggregate_only_vram(state, nodes)
        crash_forensics = [_build_crash_forensics_payload(t, state, nodes) for t in newly_crashed]
        oom_forensics = [_build_oom_forensics_payload(p, state) for p in crash_forensics
                         if _is_oom_like_forensics(p)]
        by_id = {t.get("id"): t for t in state.get("tasks", [])}
        for payload in crash_forensics:
            if payload.get("id") in by_id:
                by_id[payload["id"]]["last_crash_forensics"] = payload
        for payload in oom_forensics:
            if payload.get("id") in by_id:
                by_id[payload["id"]]["last_oom_forensics"] = payload
        # Pre-dispatch: if any GPU is over threshold from earlier dispatches, evict the youngest
        # task on it back to queue. This is the rollback companion to optimistic packing — we
        # let _reserve_inflight_vram allow stacking based on observed peak, but if the gamble
        # results in actual threshold breach, kill the latest one cleanly.
        evicted = _enforce_post_dispatch_thresholds(state, nodes)
        resource_evictions = [
            t.get("last_resource_eviction") for t in state.get("tasks", [])
            if t.get("id") in set(evicted) and t.get("last_resource_eviction")
        ]
        if evicted:
            # GPU memory will take a moment to release; re-probe so dispatch sees freed state.
            nodes = probe_all()
            _remember_running_resource_snapshots(state, nodes)
            _reconcile_aggregate_only_vram(state, nodes)
        _reserve_inflight_vram(state, nodes)
        events, qcount = _do_dispatch(state, nodes)
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
        notify("task_done", _compact_task_event_payload(t))
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
    for payload in crash_forensics:
        notify("crash_forensics", payload, feishu_enabled=False)
    for payload in oom_forensics:
        notify("oom_forensics", payload, feishu_enabled=False)
    for payload in resource_evictions:
        notify("resource_pressure_evicted", payload, feishu_enabled=False)
    for ev in events:
        if ev["type"] == "launched":
            notify("task_launched", _compact_task_event_payload(ev["task"]))
        elif ev["type"] in ("pre_launch_snapshot", "post_launch_snapshot"):
            notify(ev["type"], ev.get("snapshot") or {}, feishu_enabled=False)
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
    notify("dispatch_cycle", _dispatch_cycle_log_payload(state, nodes, events, qcount),
           feishu_enabled=False)
    # Result sync is deliberately after dispatch. A stale multi-GB result pull
    # should not keep newly queued tasks idle when nodes are available.
    try:
        _sync_completed_results_outside_lock()
    except Exception as _e:
        notify("result_sync_error_outer", {"error": str(_e)[:200]},
               feishu_enabled=False)
    for t in auto_adopted:
        notify("task_auto_adopted", _compact_task_event_payload(t))
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
        active_claim_ids_by_node = {}
        claim_records_by_node = {}
        # Phase 3.4.9 P1: also collect (task_id -> live PID) per node for
        # claim pid reconcile. update_pid in launch / orphan-adopt is best-
        # effort with try/except; if those failed (transport blip, claims
        # disabled momentarily, etc.) the claim sits at pid=None forever
        # and gets double-counted in pending-claim folding (line ~1305).
        # The watcher reconciles each cycle so the worst-case window is
        # one watch interval, not the task's lifetime.
        live_pid_by_node = {}
        for t in cur_state.get("tasks", []):
            node = t.get("node")
            if node and t.get("status") in ("running", "launching") and _ClaimManager.enabled_for(node):
                active_claim_ids_by_node.setdefault(node, set()).add(t["id"])
            if t.get("status") != "running":
                continue
            if node and _ClaimManager.enabled_for(node):
                running_by_node.setdefault(node, []).append(t["id"])
                claim_records_by_node.setdefault(node, {})[t["id"]] = (
                    _claim_resource_record_for_task(t)
                )
                pids = t.get("remote_pids") or []
                if pids:
                    live_pid_by_node.setdefault(node, {})[t["id"]] = int(pids[0])
        own_sid = _ClaimManager.scheduler_id()
        for node in NODES:
            if not _ClaimManager.enabled_for(node):
                continue
            try:
                # Release stale claims that still belong to this scheduler but
                # are no longer represented by a local running/launching task.
                # TTL-based GC intentionally keeps dead-PID claims alive until
                # expiry; without this reconciliation, a queued task whose node
                # was cleared can reserve phantom CPU/VRAM for up to claim_ttl_s.
                stale_released = []
                active_ids = active_claim_ids_by_node.get(node, set())
                try:
                    current_claims = _ClaimManager.enumerate(node)
                except Exception:
                    current_claims = []
                for claim in current_claims:
                    if claim.get("scheduler_id") != own_sid:
                        continue
                    tid = claim.get("task_id")
                    if not tid or tid in active_ids:
                        continue
                    try:
                        if _ClaimManager.release(node, tid):
                            stale_released.append(tid)
                    except Exception:
                        pass
                if stale_released:
                    notify("claims_released_inactive",
                           {"node": node, "task_ids": stale_released[:50],
                            "count": len(stale_released)},
                           feishu_enabled=False)
                ids = running_by_node.get(node, [])
                if ids:
                    _ClaimManager.renew_many(
                        node, ids, claim_records_by_node.get(node, {}))
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
    if not isinstance(me, dict):
        me = {}
    me.setdefault("pid", os.getpid())
    me.setdefault("started_at", now)
    me.setdefault("interval", getattr(args, "interval", 60))
    me.setdefault("heartbeat", getattr(args, "heartbeat", 3600))
    me.setdefault("resource_log_interval", getattr(args, "resource_log_interval", RESOURCE_LOG_INTERVAL_S))
    resource_interval = max(0, int(getattr(args, "resource_log_interval", RESOURCE_LOG_INTERVAL_S) or 0))
    me_dirty = False
    if resource_interval and now - me.get("last_resource_log_ts", 0) >= resource_interval:
        with state_lock():
            state = load_state()
        notify("node_cpu_accounting", _resource_accounting_payload(state, nodes),
               feishu_enabled=False)
        me["last_resource_log_ts"] = now
        me_dirty = True
    if now - me.get("last_heartbeat_ts", 0) >= args.heartbeat:
        with state_lock():
            state = load_state()
        payload = _build_heartbeat_payload(state, nodes)
        notify("heartbeat", payload)
        me["last_heartbeat_ts"] = now
        me_dirty = True
    if me_dirty:
        try: WATCHER_STATE.write_text(json.dumps(me))
        except Exception: pass

def _format_task_vram_usage(task):
    if task.get("status") == "running":
        cur = int(task.get("current_vram_mb") or 0)
        if cur > 0:
            return f"cur={_format_mem_gb(cur)}"
        if task.get("vram_estimation_source") == "aggregate_observed_zero":
            return "cur=0.00GB"
    if task.get("peak_vram_mb"):
        return f"peak={_format_mem_gb(task['peak_vram_mb'])}"
    return _format_mem_gb(task.get("est_vram_mb", 0), approx=True)


def _format_task_ram_usage(task):
    if task.get("status") == "running":
        cur = int(task.get("current_ram_mb") or 0)
        if cur > 0:
            return f"Rcur={_format_mem_gb(cur)}"
    if task.get("peak_ram_mb"):
        return f"Rpeak={_format_mem_gb(task['peak_ram_mb'])}"
    return _format_mem_gb(task.get("ram_mb", 0), approx=True)


def _task_result_artifacts_for_display(task: dict, include_log: bool = True) -> list:
    stored = task.get("result_artifacts") or []
    if stored:
        return stored
    return _discover_result_artifacts(task, include_log=include_log)


def _print_result_artifacts(task: dict, include_log: bool = True):
    artifacts = _task_result_artifacts_for_display(task, include_log=include_log)
    if not artifacts:
        return
    print("\n# result artifacts:")
    for rec in artifacts:
        node = rec.get("node") or task.get("node") or "unknown"
        kind = rec.get("kind") or "path"
        src = rec.get("source") or "inferred"
        path = _remote_path_for_node(node, rec.get("path")) if node in NODES else rec.get("path")
        print(f"#   - [{kind}] {node}:{path}  ({src})")


def cmd_status(args):
    with state_lock():
        state = load_state()
        recover_stale_launching_tasks(state)
        update_running_tasks(state)
        _seed_pending_eta_from_history(state)
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
        if n.get("slurm_cluster"):
            etaload = node_loads.get(n["name"], 0)
            if etaload <= 0:
                etaload_str = ""
            elif etaload < 3600:
                etaload_str = f"  eta_load={etaload/60:.0f}m"
            elif etaload < 86400:
                etaload_str = f"  eta_load={etaload/3600:.1f}h"
            else:
                etaload_str = f"  eta_load={etaload/86400:.1f}d"
            print(f"  {n['name']:11s} {_format_slurm_cluster_summary(n)}{etaload_str}")
            continue
        gpu_parts = []
        for g in n["gpus"]:
            mem_pct = int(round(100 * g["used_mb"] / max(g["total_mb"], 1)))
            gpu_parts.append(
                f"GPU{g['idx']}={_format_mem_gb(g['used_mb'])}/{_format_mem_gb(g['total_mb'])}"
                f"(free={_format_mem_gb(g.get('free_mb', 0))}, mem:{mem_pct}%, util:{g['util_pct']}%)")
        gpu_str = ", ".join(gpu_parts)
        load = n.get("loadavg", 0)
        reserve = int(NODES.get(n.get("name"), {}).get("reserved_cpu_cores") or 0)
        reserve_str = f", reserve {reserve}" if reserve else ""
        cpu_str = f"cpu={n.get('free_cpu', '?')}/{n.get('total_cpu', '?')}(load {load:.1f}{reserve_str})"
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
        ram_str = _format_node_ram_summary(n)
        print(f"  {n['name']:11s} {gpu_str}  {cpu_str}  {ram_str}{etaload_str}{claim_str}")
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
        eta = _format_task_eta(t)
        eta = f" {eta:15s}" if eta else " " * 16
        proj = (t.get("project") or "?")[:14]
        owner = _format_task_owner(t)[:18]
        print(f"  [{t['id']}] {t['status']:9s} {loc:20s} {proj:14s} {owner:18s} {peak:14s} {pram:13s}{runtime}{eta} {t['description'][:55]}")


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
    with state_lock():
        t, source = _find_task_record(args.id, include_archive=True)
    if not t:
        sys.exit(f"task {args.id} not found in queue or archive")
    print(json.dumps(t, indent=2))
    if source == "archive":
        print("\n# source: archive (terminal task moved out of hot queue.json)")
    if t.get("log_path") and t.get("node"):
        host = NODES.get(t["node"], {}).get("host") or "local"
        if _node_is_windows(t["node"]):
            cmd = _ssh_base_args(t["node"]) + [
                "powershell", "Get-Content", "-Tail", "80", "-Wait",
                "-LiteralPath", t["log_path"],
            ]
            print(f"\n# tail log: {' '.join(shlex.quote(a) for a in cmd)}")
        elif host == "local":
            print(f"\n# tail log: tail -f {shlex.quote(t['log_path'])}")
        else:
            cmd = _ssh_base_args(t["node"]) + [f"tail -f {shlex.quote(t['log_path'])}"]
            print(f"\n# tail log: {' '.join(shlex.quote(a) for a in cmd)}")
    _print_result_artifacts(t, include_log=True)


def cmd_results(args):
    """Find result artifacts for tasks in queue + archive."""
    import fnmatch as _fnmatch

    task_ids = set(args.task_ids or [])
    include_archive = not args.no_archive
    scan_logs = (bool(task_ids) or bool(args.scan_logs)) and not args.no_log_scan
    with state_lock():
        state = load_state()
        records = [("queue", t) for t in state.get("tasks", [])]
        if include_archive:
            records.extend(("archive", t) for t in load_archive_tasks())

    statuses = set(args.status or [])
    candidates = []
    for source, t in records:
        if task_ids and t.get("id") not in task_ids:
            continue
        if args.project and not _fnmatch.fnmatch(t.get("project") or "", args.project):
            continue
        if args.signature and not _fnmatch.fnmatch(t.get("signature") or "", args.signature):
            continue
        if statuses and t.get("status") not in statuses:
            continue
        candidates.append((source, t))

    candidates.sort(key=lambda it: float(
        it[1].get("finished_at") or it[1].get("started_at") or it[1].get("submitted_at") or 0
    ), reverse=True)

    rows = []
    scanned = 0
    for source, t in candidates:
        if args.limit and len(rows) >= args.limit:
            break
        scanned += 1
        arts = _task_result_artifacts_for_display(t, include_log=scan_logs)
        if not arts and not args.include_empty and not task_ids:
            continue
        rows.append({
            "source": source,
            "id": t.get("id"),
            "status": t.get("status"),
            "project": t.get("project"),
            "signature": t.get("signature"),
            "node": t.get("node"),
            "description": t.get("description"),
            "log_path": t.get("log_path"),
            "artifacts": arts,
        })

    if args.json:
        print(json.dumps({"results": rows, "matched": len(candidates), "scanned": scanned}, indent=2))
        return
    if not rows:
        where = "queue+archive" if include_archive else "queue"
        print(f"(no result artifacts found in {where}; matched {len(candidates)} task records)")
        if not scan_logs:
            print("  note: logs were not scanned for this batch query; add --scan-logs or pass task IDs")
        return
    for r in rows:
        desc = (r.get("description") or "")[:80]
        print(f"[{r['id']}] {r['status']} {r.get('project') or '?'} "
              f"{r.get('node') or '-'} ({r['source']})  {desc}")
        if r.get("signature"):
            print(f"  sig: {r['signature']}")
        if r.get("log_path"):
            print(f"  log: {r['log_path']}")
        arts = r.get("artifacts") or []
        if not arts:
            print("  results: (none inferred)")
        for rec in arts:
            print(f"  - [{rec.get('kind') or 'path'}] {rec.get('node') or r.get('node') or 'unknown'}:"
                  f"{rec.get('path')}  ({rec.get('source') or 'inferred'})")

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
                notify("task_cancelled", {
                    "task_id": t.get("id"),
                    "status_before": prev,
                    "actor": t.get("cancel_actor"),
                    "reason": t.get("cancel_reason"),
                    "related_cancelled": related,
                }, feishu_enabled=False)
                suffix = f" (+{related} duplicate queued retry)" if related else ""
                print(f"cancelled {prev} task {args.id}{suffix} by {t.get('cancelled_by')}")
                return
            if t["status"] == "running":
                if not args.force:
                    sys.exit(f"task {args.id} is RUNNING — pass --force to kill it (will not affect other tasks)")
                pids = _task_pids(t)
                actor = _record_task_kill_actor(t, "user force-cancel", "scheduler.py cancel --force")
                ok, kill_msg = _kill_task_processes(t, timeout=15)
                # Phase 3.2.1: release the cross-scheduler claim too. Best-
                # effort; failure leaves the claim to expire via TTL + GC.
                try:
                    _release_task_claims_and_intents(t)
                except Exception:
                    pass
                _mark_user_cancelled(t, "user force-cancel")
                t["last_kill_actor"] = actor
                t["last_killed_by"] = actor["label"]
                related = _cancel_related_queued_retries(state, t, "user force-cancel")
                save_state(state)
                notify("task_killed", {
                    "task_id": t.get("id"),
                    "node": t.get("node"),
                    "pids": pids,
                    "actor": actor,
                    "action": "user force-cancel",
                    "reason": "scheduler.py cancel --force",
                    "kill_ok": ok,
                    "kill_msg": kill_msg,
                    "related_cancelled": related,
                }, feishu_enabled=False)
                suffix = kill_msg if ok else f"kill warning: {kill_msg}"
                dup_suffix = f"; also cancelled {related} duplicate queued retry" if related else ""
                print(f"killed pids={pids} on {t['node']} by {actor['label']} and cancelled {args.id} ({suffix}{dup_suffix})")
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
        notify("clear_queue_cancelled", {
            "count": len(ids),
            "task_ids": ids,
            "actor": _actor_info("clear-queue", "user clear-queue"),
        }, feishu_enabled=False)
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
            _clear_live_eta_fields(t, clear_runtime_projection=True)
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
    if _node_is_windows(args.node):
        sys.exit("adopt is Linux/GPU-only for now; Windows jtl110cpu tasks must be launched through scheduler.py")
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
            "origin": "manual-adopt",
            "submitted_by": _local_user(),
            "process_owner": _local_user(),
            "submitted_host": args.node,
            "scheduler_id": None,
            "shared_account_suspect": True,
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
            "require_node", "require_gpu_idx", "allow_gpu_over_one_third")):
        sys.exit("specify at least one of --vram-mb / --ram-mb / --cpu / "
                 "--description / --preferred-node / --require-node / --require-gpu / "
                 "--allow-gpu-over-one-third")
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
                t["est_vram_mb_explicit"] = True
            if args.ram_mb is not None:
                changes.append(("ram_mb", t.get("ram_mb"), int(args.ram_mb)))
                t["ram_mb"] = int(args.ram_mb)
                t["ram_mb_explicit"] = True
            if args.cpu is not None:
                changes.append(("cpu_cores", t.get("cpu_cores"), int(args.cpu)))
                t["cpu_cores"] = int(args.cpu)
                t["cpu_cores_explicit"] = True
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
            if args.require_gpu_idx is not None:
                raw = str(args.require_gpu_idx).strip()
                if raw == "":
                    new_gpu = None
                else:
                    try:
                        new_gpu = int(raw)
                    except ValueError:
                        sys.exit(f"--require-gpu expects a non-negative integer or empty string, got {raw!r}")
                    if new_gpu < 0:
                        sys.exit(f"--require-gpu expects a non-negative integer or empty string, got {raw!r}")
                changes.append(("require_gpu_idx", t.get("require_gpu_idx"), new_gpu))
                if new_gpu is None:
                    t.pop("require_gpu_idx", None)
                else:
                    t["require_gpu_idx"] = new_gpu
                placement_changed = True
            if args.allow_gpu_over_one_third is not None:
                new_allow = bool(args.allow_gpu_over_one_third)
                changes.append(("allow_gpu_over_one_third",
                                t.get("allow_gpu_over_one_third"), new_allow))
                if new_allow:
                    t["allow_gpu_over_one_third"] = True
                else:
                    t.pop("allow_gpu_over_one_third", None)
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
    if not _requires_local_capacity_check(name, task, node_state):
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
    cpu_fallback_node = (
        _node_is_windows(name)
        or not node_state.get("gpus")
        or node_info.get("max_vram_per_task") == 0
    )
    if cpu_fallback_node:
        block = _node_cpu_fallback_block_reason(task, name, node_info)
        if block:
            return f"BLOCKED: {block}{soft_note}"
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
        algo_reason = _algorithm_gpu_fit_block_reason(task, g, node_info)
        if algo_reason:
            reasons.append(algo_reason)
        if cap_per_task is not None and est > cap_per_task:
            reasons.append(f"est={est}>per-task cap={cap_per_task}")
        freeze = _gpu_freeze_line_mb(int(g.get("total_mb") or 0))
        if (not _task_ignores_one_third_pack_rule(task, node_info)
                and freeze > 0 and g["used_mb"] > 100
                and (g["used_mb"] >= freeze or g["used_mb"] + est >= freeze)):
            reasons.append(f"used+est {g['used_mb']}+{est}/{g['total_mb']}MB ≥ 1/3+grace {freeze}MB (packing rule)")
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
    gpu_pin = task.get("require_gpu_idx")
    gpu_pin_text = "" if gpu_pin is None else f"  require_gpu: {gpu_pin!r}"
    print(f"  preferred:    {task.get('preferred_node')!r}  require:    {task.get('require_node')!r}{gpu_pin_text}")
    print(f"  est:          vram={task.get('est_vram_mb')}MB ram={task.get('ram_mb')}MB cpu={task.get('cpu_cores')}")
    if task.get("wait_for_files"):
        print(f"  wait_files:   {task.get('wait_for_files')}")
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
    s.add_argument("--cpu-parallel-items", dest="cpu_parallel_items", type=int, default=0,
                   help="For CPU-only multi-worker jobs with M independent items, auto-size workers as "
                        "e=ceil(M/physical_cores), workers=ceil(M/e). Commands may use "
                        "{workers}/{n_workers}/{num_workers} placeholders or pass --workers auto.")
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
    s.add_argument("--wait-for-file", dest="wait_for_files", action="append",
                   help="Defer dispatch until this local prerequisite file exists and is non-empty. "
                        "Repeat for multiple prerequisites; useful for eval tasks waiting on train ckpts.")
    s.add_argument("--test-log", dest="test_log",
                   help="Local preflight/test log containing tqdm/progress output. "
                        "Parsed at submit time and recorded into runtime history so ETA/walltime "
                        "use the local test profile before the real experiment launches.")
    s.add_argument("--test-peak-vram-mb", dest="test_peak_vram_mb", type=int,
                   help="Peak VRAM observed during local preflight. Recorded into signature history before sizing this submit.")
    s.add_argument("--test-peak-ram-mb", dest="test_peak_ram_mb", type=int,
                   help="Peak RAM observed during local preflight. Recorded into signature history before sizing this submit.")
    s.add_argument("--test-cpu", dest="test_cpu", type=int,
                   help="CPU cores observed during local preflight. Recorded into signature history before sizing this submit.")
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
    s.add_argument("--allow-remote-large-data", dest="allow_remote_large_data", action="store_true",
                   help="Override SimpleSAC large external-data local pin. Use only after manually "
                        "staging snapshot/per-policy data on the target node.")
    s.add_argument("--allow-duplicate", dest="allow_duplicate", action="store_true",
                   help="Allow submission even when run identity matches an existing queued/launching/running task")
    s.add_argument("--allow-seed-batch", dest="allow_seed_batch", action="store_true",
                   help="Do not auto-split simple BAPR `for seed in ...` shell loops at submit time.")
    s.add_argument("--stage-exclude", dest="stage_excludes", action="append",
                   help="Extra cwd staging exclude path/glob, relative to --cwd. Repeatable.")
    s.add_argument("--reroute-on-node-down", dest="reroute_on_node_down", action="store_true",
                   help="Opt into auto-requeue/reroute if a running task's node probe stays unknown.")
    s.add_argument("--node-down-requeue-s", dest="node_down_requeue_s", type=int, default=0,
                   help=f"Seconds of unknown node probe before reroute when opted in (default {NODE_DOWN_REQUEUE_S}).")
    s.add_argument("--slurm-partition", dest="slurm_partition", default="",
                   help="Optional Slurm partition to pass as #SBATCH --partition when routed to SlurmBackend")
    s.add_argument("--slurm-account", dest="slurm_account", default="",
                   help="Optional Slurm account to pass as #SBATCH --account when routed to SlurmBackend")
    s.add_argument("--slurm-qos", dest="slurm_qos", default="",
                   help="Optional Slurm QoS to pass as #SBATCH --qos when routed to SlurmBackend")
    s.set_defaults(func=cmd_submit)

    s = sub.add_parser("cpu-plan", help="Plan M independent CPU items across physical-core CPU nodes")
    s.add_argument("--items", type=int, required=True,
                   help="Logical CPU items/checkpoints to process")
    s.add_argument("--item-multiplier", type=int, default=1,
                   help="Independent work items per logical item. Example: 39 ckpts * "
                        "10 eval episodes => --items 39 --item-multiplier 10.")
    s.add_argument("--nodes", default="", help="Comma-separated CPU nodes (default: all cpu_labor_node nodes)")
    s.add_argument("--use-total-cores", action="store_true",
                   help="Ignore live free_cpu and plan from full physical-core capacity")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_cpu_plan)

    s = sub.add_parser("submit-cpu-batch",
                       help="Split a CPU-heavy batch across CPU nodes and submit one shard per node")
    s.add_argument("--items", type=int, required=True,
                   help="Logical CPU items/checkpoints to process")
    s.add_argument("--item-multiplier", type=int, default=1,
                   help="Independent work items per logical item. Example: ckpt eval with "
                        "10 episodes per checkpoint should use --item-multiplier 10 so "
                        "worker planning sees ckpt_count*10 work items.")
    s.add_argument("--cmd-template", required=True,
                   help="Command template. Known placeholders: {start} {end} {items} "
                        "{total_items} {logical_items} {item_multiplier} "
                        "{workers} {node} {shard_index} {num_shards}. "
                        "Use {workers} or '--workers auto' to get the computed worker count.")
    s.add_argument("--cwd", required=True,
                   help="Working directory template/path on target node")
    s.add_argument("--signature", required=True,
                   help="Signature template, e.g. Project/eval/{node}")
    s.add_argument("--description", required=True,
                   help="Description template")
    s.add_argument("--nodes", default="",
                   help="Comma-separated CPU nodes (default: all cpu_labor_node nodes)")
    s.add_argument("--use-total-cores", action="store_true",
                   help="Ignore live free_cpu and plan from full physical-core capacity")
    s.add_argument("--ram-mb", dest="ram_mb", type=int,
                   help="RAM estimate per shard")
    s.add_argument("--priority", choices=["low", "normal", "high"], default="normal")
    s.add_argument("--project", help="Project name")
    s.add_argument("--env", nargs="*", help="Extra env vars KEY=VALUE")
    s.add_argument("--result-dir-template", dest="result_dir_template",
                   help="Remote result dir template for each shard")
    s.add_argument("--local-result-dir-template", dest="local_result_dir_template",
                   help="Local result dir template for each shard")
    s.add_argument("--wait-for-file-template", dest="wait_for_file_template", action="append",
                   help="Prerequisite file template; repeatable")
    s.add_argument("--allow-env-only-shard", action="store_true",
                   help="Allow multi-node split when templates lack {start}/{end}/{node}; "
                        "use only if the script reads SCHEDULEURM_CPU_* env vars.")
    s.add_argument("--allow-cpu-training", dest="allow_cpu_training", action="store_true")
    s.add_argument("--cpu-training-justification", dest="cpu_training_justification", default="")
    s.add_argument("--allow-no-ckpt", dest="allow_no_ckpt", action="store_true")
    s.add_argument("--allow-no-resume", dest="allow_no_resume", action="store_true")
    s.add_argument("--allow-shared-result-dir", dest="allow_shared_result_dir", action="store_true")
    s.add_argument("--allow-remote-large-data", dest="allow_remote_large_data", action="store_true")
    s.add_argument("--allow-duplicate", dest="allow_duplicate", action="store_true")
    s.add_argument("--stage-exclude", dest="stage_exclude", action="append",
                   help="Extra cwd staging exclude path/glob, relative to --cwd. Repeatable.")
    s.add_argument("--node-down-requeue-s", dest="node_down_requeue_s", type=int, default=0,
                   help=f"Seconds of unknown node probe before CPU batch shard reroute (default {NODE_DOWN_REQUEUE_S}).")
    s.add_argument("--env-spec", dest="env_spec", default="none")
    s.add_argument("--image", dest="image", default="")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_submit_cpu_batch)

    s = sub.add_parser("dispatch", help="Probe nodes & launch what fits (also rebalances queue)")
    s.add_argument("--algorithm", default="",
                   help="Placement algorithm policy (default/env: legacy). Examples: legacy, sweetspot_v1")
    s.set_defaults(func=cmd_dispatch)

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
    s.add_argument("--resource-log-interval", dest="resource_log_interval", type=int,
                   default=RESOURCE_LOG_INTERVAL_S,
                   help=f"Write detailed node CPU/RAM attribution to watcher.log every N seconds (default {RESOURCE_LOG_INTERVAL_S})")
    s.add_argument("--algorithm", default="",
                   help="Placement algorithm policy (default/env: legacy). Examples: legacy, sweetspot_v1")
    s.set_defaults(func=cmd_watch)

    s = sub.add_parser("status", help="Show node + task state")
    s.add_argument("--all", action="store_true", help="Include done/failed/cancelled tasks")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("doctor", help="Audit active queue invariants; --fix applies safe queued-task repairs")
    s.add_argument("--fix", action="store_true",
                   help="Apply safe repairs to queued tasks: add wait-for-file gates, force SimpleSAC large-data local, promote dependent train priority")
    s.add_argument("--project", help="fnmatch glob over project (e.g. SimpleSAC)")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_doctor)

    s = sub.add_parser("profile-local", help="Run a local preflight directly, monitor peak RAM/VRAM/CPU, and record tqdm/runtime history")
    s.add_argument("--description", default="local preflight profile")
    s.add_argument("--cmd", required=True, help="Shell command to run locally (not through scheduleurm queue)")
    s.add_argument("--cwd", required=True, help="Working directory for the local preflight")
    s.add_argument("--signature", required=True, help="Signature whose resource/runtime history should be updated")
    s.add_argument("--project", help="Project name (else derived from cwd/signature)")
    s.add_argument("--env", nargs="*", help="Extra env vars KEY=VALUE")
    s.add_argument("--log-path", dest="log_path", help="Where to write the local preflight log")
    s.add_argument("--sample-interval", dest="sample_interval", type=float, default=5.0,
                   help="Seconds between local resource samples (default 5)")
    s.add_argument("--timeout", type=int, default=0,
                   help="Kill the local preflight after N seconds (0 = no timeout)")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_profile_local)

    s = sub.add_parser("claims", help="Show remote shared claims/intents for claims-enabled nodes")
    s.add_argument("--node", choices=list(NODES.keys()), help="Limit to one node")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_claims)

    s = sub.add_parser("show", help="Show one task's full record + how to tail logs")
    s.add_argument("id"); s.set_defaults(func=cmd_show)

    s = sub.add_parser("results", help="Find inferred result artifacts in queue + archive")
    s.add_argument("task_ids", nargs="*", help="Optional task IDs to inspect")
    s.add_argument("--project", help="fnmatch glob over project, e.g. 'SimpleSAC' or 'bapr*'")
    s.add_argument("--signature", help="fnmatch glob over signature, e.g. 'H2Oplus/r3_eval_*'")
    s.add_argument("--status", nargs="*", default=["done"],
                   help="Statuses to include (default: done). Pass e.g. --status done failed")
    s.add_argument("--limit", type=int, default=50, help="Max rows to print (default 50; 0 = no limit)")
    s.add_argument("--no-archive", action="store_true", help="Only search hot queue.json")
    s.add_argument("--scan-logs", action="store_true",
                   help="For batch queries, scan task logs for saved output paths. "
                        "Task-id queries scan logs by default.")
    s.add_argument("--no-log-scan", action="store_true",
                   help="Do not read task logs; use stored fields and command flags only")
    s.add_argument("--include-empty", action="store_true",
                   help="Print matching tasks even when no result artifact can be inferred")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_results)

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
    s.add_argument("--require-gpu", dest="require_gpu_idx",
                   help="Set / change hard GPU index pin on the selected node (use empty string to clear)")
    s.add_argument("--allow-gpu-over-one-third", dest="allow_gpu_over_one_third",
                   action="store_true", default=None,
                   help="Allow this queued GPU task to exceed the 1/3 GPU packing guard")
    s.add_argument("--clear-gpu-over-one-third", dest="allow_gpu_over_one_third",
                   action="store_false", default=None,
                   help="Clear the per-task 1/3 GPU packing override")
    s.set_defaults(func=cmd_edit)

    s = sub.add_parser("why", help="Diagnose why a queued task isn't being dispatched")
    s.add_argument("id", help="Task id (e.g. t0042)")
    s.set_defaults(func=cmd_why)

    sub.add_parser("tui", help="Interactive TUI: sortable + filterable + auto-refresh task table").set_defaults(func=cmd_tui)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
