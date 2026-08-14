"""Import-time scheduler configuration and runtime-global defaults."""

from __future__ import annotations

import os
from pathlib import Path


def _env_bool(environ: dict, key: str, default: str) -> bool:
    return str(environ.get(key, default)).lower() not in ("0", "false", "no", "off")


def _env_int(environ: dict, key: str, default: str) -> int:
    return int(environ.get(key, default))


def _env_float(environ: dict, key: str, default: str) -> float:
    return float(environ.get(key, default))


def build_scheduler_config_exports(*, environ=None, home=None) -> dict:
    env = os.environ if environ is None else environ
    home_path = Path.home() if home is None else Path(home)
    state_dir = home_path / ".claude" / "scheduler"
    defer_launch_outside_lock = _env_bool(env, "SCHEDULEURM_DEFER_LAUNCH_OUTSIDE_LOCK", "1")
    migration_min_task_eta_s = _env_int(env, "SCHEDULEURM_MIGRATION_MIN_TASK_ETA_S", "300")
    result_sync_timeout_s = _env_int(env, "SCHEDULEURM_RESULT_SYNC_TIMEOUT_S", "1800")
    return {
        "SCHEDULER_CONTROL_SUBCOMMANDS": {
            "adopt",
            "cancel",
            "cancel-batch",
            "claims",
            "clear-queue",
            "compact",
            "dispatch",
            "doctor",
            "forget",
            "history",
            "log",
            "profile-local",
            "record-vram",
            "repair-queued",
            "results",
            "show",
            "status",
            "submit",
            "submit-jsonl",
            "submit-cpu-batch",
            "task-log",
            "wait-for",
            "watch",
        },
        "SCHEDULER_CONTROL_PATH_MARKERS": (
            "/.claude/skills/scheduler/",
            "/scheduleurm/skill/",
        ),
        "STATE_DIR": state_dir,
        "QUEUE_FILE": state_dir / "queue.json",
        "VRAM_FILE": state_dir / "vram_history.json",
        "RUNTIME_FILE": state_dir / "runtime_history.json",
        "NODE_PROBE_CACHE_FILE": state_dir / "node_probe_cache.json",
        "LOCK_FILE": state_dir / ".lock",
        "HISTORY_LOCK_FILE": state_dir / ".history.lock",
        "RESULT_SYNC_WORKER_LOCK_FILE": state_dir / ".result_sync_worker.lock",
        "WATCHER_LOCK_FILE": state_dir / ".watcher.lock",
        "DISPATCH_INTENT_FILE": state_dir / ".dispatch_intent.json",
        "LOG_DIR": state_dir / "logs",
        "ETA_ANALYSIS_DIR": state_dir / "logs" / "eta_analysis",
        "ALLOW_EMPTY_QUEUE_RESET_ENV": "SCHEDULEURM_ALLOW_EMPTY_QUEUE_RESET",
        "EMPTY_QUEUE_RESET_GUARD_MIN_TASKS": 20,
        "EMPTY_QUEUE_RESET_GUARD_MIN_ACTIVE": 5,
        "QUEUE_MISSING_GUARD_MAX_AGE_S": 6 * 3600,
        "VRAM_MARGIN_MB": 500,
        "RAM_HEADROOM_FRAC": 0.10,
        "ONE_THIRD_PACK_RULE": True,
        "ONE_THIRD_PACK_GRACE_MB": _env_int(env, "SCHEDULEURM_ONE_THIRD_PACK_GRACE_MB", "512"),
        "GPU_EVICT_ROLLBACK_MAX_AGE_S": _env_int(env, "SCHEDULEURM_GPU_EVICT_ROLLBACK_MAX_AGE_S", "1800"),
        "GPU_EVICT_STABLE_PROGRESS_MIN_AGE_S": _env_int(env, "SCHEDULEURM_GPU_EVICT_STABLE_PROGRESS_MIN_AGE_S", "900"),
        "RAM_HEADROOM_EVICTION_GRACE_MB": _env_int(env, "SCHEDULEURM_RAM_HEADROOM_EVICTION_GRACE_MB", "512"),
        "RAM_EVICT_CKPT_PROTECT_MIN_AGE_S": _env_int(env, "SCHEDULEURM_RAM_EVICT_CKPT_PROTECT_MIN_AGE_S", "300"),
        "RAM_EVICT_CKPT_PROTECT_PROGRESS": _env_float(env, "SCHEDULEURM_RAM_EVICT_CKPT_PROTECT_PROGRESS", "0.01"),
        "GPU_UTIL_SATURATION_PCT": 85,
        "DEFAULT_VRAM_MB": 512,
        "GPU_EMPTY_USED_MB": 200,
        "DEFAULT_RAM_MB": 4096,
        "DEFAULT_CPU_CORES": 1,
        "DISPATCH_INTENT_TTL_S": _env_float(env, "SCHEDULEURM_DISPATCH_INTENT_TTL_S", "900"),
        # Bulk submit is atomic under state_lock, and deferred launches are
        # protected by launching state plus per-node launch slots.  Pausing the
        # whole watcher here only delays terminal reconciliation and the next
        # dispatch wave; retain the switch solely for legacy deployments.
        "WATCH_RESPECTS_DISPATCH_INTENT": _env_bool(env, "SCHEDULEURM_WATCH_RESPECTS_DISPATCH_INTENT", "0"),
        "CANCEL_DEFERS_TO_DISPATCH_INTENT": _env_bool(env, "SCHEDULEURM_CANCEL_DEFERS_TO_DISPATCH_INTENT", "1"),
        "STATUS_READONLY_LOCK_TIMEOUT_S": _env_float(env, "SCHEDULEURM_STATUS_READONLY_LOCK_TIMEOUT_S", "2.0"),
        "STATE_LOCK_WARN_WAIT_S": _env_float(env, "SCHEDULEURM_STATE_LOCK_WARN_WAIT_S", "2.0"),
        "STATE_LOCK_WARN_HOLD_S": _env_float(env, "SCHEDULEURM_STATE_LOCK_WARN_HOLD_S", "5.0"),
        "AUTO_ADOPT_INTERVAL_S": max(0, _env_int(env, "SCHEDULEURM_AUTO_ADOPT_INTERVAL_S", "60")),
        "ARCHIVE_FILE": state_dir / "queue_archive.jsonl",
        "ARCHIVE_AGE_DAYS": _env_float(env, "SCHEDULEURM_ARCHIVE_AGE_DAYS", "1"),
        "ARCHIVE_MAX_HOT_TERMINAL": _env_int(env, "SCHEDULEURM_ARCHIVE_MAX_HOT_TERMINAL", "500"),
        "WATCHER_LOG_MAX_MB": 50,
        "WATCHER_LOG_GENERATIONS": 3,
        "RESOURCE_LOG_INTERVAL_S": max(30, _env_int(env, "SCHEDULEURM_RESOURCE_LOG_INTERVAL_S", "300")),
        "WINDOWS_WRAPPER_RESOURCE_LOG_INTERVAL_S": max(
            10, _env_int(env, "SCHEDULEURM_WINDOWS_WRAPPER_RESOURCE_LOG_INTERVAL_S", "60")
        ),
        "RUNNING_PROBE_MIN_INTERVAL_S": max(0, _env_int(env, "SCHEDULEURM_RUNNING_PROBE_MIN_INTERVAL_S", "30")),
        "RUNNING_PROBE_MAX_WORKERS": max(1, _env_int(env, "SCHEDULEURM_RUNNING_PROBE_MAX_WORKERS", "8")),
        "RUNNING_PROBE_TIMEOUT_S": max(1, _env_int(env, "SCHEDULEURM_RUNNING_PROBE_TIMEOUT_S", "25")),
        "CANCEL_KILL_MAX_WORKERS": max(1, _env_int(env, "SCHEDULEURM_CANCEL_KILL_MAX_WORKERS", "4")),
        # A single watcher pass must be able to expose enough one-core work to
        # fill all six 192-core CPU nodes. Placement debits resources
        # immediately, so this is a scan/throughput limit rather than an
        # overcommit limit.
        "WATCH_DISPATCH_MAX_QUEUED_PER_CYCLE": max(
            0, _env_int(env, "SCHEDULEURM_WATCH_DISPATCH_MAX_QUEUED_PER_CYCLE", "1024")
        ),
        "TERMINAL_DIAGNOSTICS_MAX_PER_CYCLE": max(
            0, _env_int(env, "SCHEDULEURM_TERMINAL_DIAGNOSTICS_MAX_PER_CYCLE", "8")
        ),
        "TERMINAL_DIAGNOSTICS_MAX_WALL_S": max(
            0.0, _env_float(env, "SCHEDULEURM_TERMINAL_DIAGNOSTICS_MAX_WALL_S", "20")
        ),
        "TERMINAL_DIAGNOSTICS_INCLUDE_LOG_ARTIFACTS": _env_bool(
            env, "SCHEDULEURM_TERMINAL_DIAGNOSTICS_INCLUDE_LOG_ARTIFACTS", "0"
        ),
        "TERMINAL_EVIDENCE_MAX_WORKERS": max(
            1, _env_int(env, "SCHEDULEURM_TERMINAL_EVIDENCE_MAX_WORKERS", "4")
        ),
        "TERMINAL_EVIDENCE_BATCH_SIZE": max(
            1, _env_int(env, "SCHEDULEURM_TERMINAL_EVIDENCE_BATCH_SIZE", "32")
        ),
        "RECOVER_QUEUED_LIVE_REMOTE_NODES": _env_bool(
            env, "SCHEDULEURM_RECOVER_QUEUED_LIVE_REMOTE_NODES", "0"
        ),
        "ETA_REFRESH_MIN_INTERVAL_S": max(0, _env_int(env, "SCHEDULEURM_ETA_REFRESH_MIN_INTERVAL_S", "60")),
        "ETA_ANALYSIS_INTERVAL_S": max(
            1, _env_int(env, "SCHEDULEURM_ETA_ANALYSIS_INTERVAL_S", "60")
        ),
        # ETA log tails are observational. A slow/broken SSH route must not
        # postpone the state-update and dispatch phase by the generic 30s
        # command timeout (or twice that when a proxy fallback is attempted).
        "ETA_TAIL_PROBE_TIMEOUT_S": max(
            1, _env_int(env, "SCHEDULEURM_ETA_TAIL_PROBE_TIMEOUT_S", "10")
        ),
        "ETA_HISTORY_OVERRUN_FLOOR_S": max(0, _env_int(env, "SCHEDULEURM_ETA_HISTORY_OVERRUN_FLOOR_S", "1800")),
        "ETA_HISTORY_OVERRUN_MAX_S": max(0, _env_int(env, "SCHEDULEURM_ETA_HISTORY_OVERRUN_MAX_S", "7200")),
        "ETA_HISTORY_OVERRUN_FRACTION": max(0.0, _env_float(env, "SCHEDULEURM_ETA_HISTORY_OVERRUN_FRACTION", "0.25")),
        "MAX_AUTO_RETRY": 3,
        "MAX_LAUNCH_RETRY": 3,
        "LAUNCH_FAILED_NODE_TTL_S": max(0, _env_int(env, "SCHEDULEURM_LAUNCH_FAILED_NODE_TTL_S", "90")),
        "DEFER_LAUNCH_OUTSIDE_LOCK": defer_launch_outside_lock,
        "LAUNCHING_RESET_S": _env_int(
            env,
            "SCHEDULEURM_LAUNCHING_RESET_S",
            "300" if defer_launch_outside_lock else "60",
        ),
        "LAUNCH_EXEC_LOCK_FILE": state_dir / ".launch_exec.lock",
        "LAUNCH_EXEC_MAX_WORKERS": max(1, _env_int(env, "SCHEDULEURM_LAUNCH_EXEC_MAX_WORKERS", "48")),
        "LAUNCH_EXEC_MAX_PER_NODE": max(1, _env_int(env, "SCHEDULEURM_LAUNCH_EXEC_MAX_PER_NODE", "12")),
        "LAUNCH_EXEC_LOCK_TIMEOUT_S": _env_float(env, "SCHEDULEURM_LAUNCH_EXEC_LOCK_TIMEOUT_S", "900"),
        "WATCH_PHASE_WARN_S": _env_float(env, "SCHEDULEURM_WATCH_PHASE_WARN_S", "3.0"),
        "NODE_DOWN_REQUEUE_S": 300,
        "ESCALATIONS_FILE": state_dir / "escalations.jsonl",
        "WINDOWS_CPU_MAX_WORKERS_PER_PROCESS": _env_int(env, "SCHEDULEURM_WINDOWS_CPU_MAX_WORKERS_PER_PROCESS", "60"),
        "ENV_MISSING_PATTERNS": ("没有那个文件或目录", "no such file or directory", "command not found", "未找到命令"),
        "PYTHON_IMPORT_PATTERNS": (
            "ModuleNotFoundError", "ImportError", "No module named"),
        "CUDA_RUNTIME_PATTERNS": (
            "gpusolverDnCreate",
            "cuSolver internal error",
            "no supported devices found for platform CUDA",
            "Unable to initialize backend 'cuda'",
        ),
        "PROJECT_WIDE_ENV_BLOCK_TTL_S": _env_int(env, "SCHEDULEURM_PROJECT_ENV_BLOCK_TTL_S", str(6 * 3600)),
        "PROBE_ALL_MAX_WORKERS": max(1, _env_int(env, "SCHEDULEURM_PROBE_ALL_MAX_WORKERS", "8")),
        "PROBE_ROUTE_MAX_WORKERS": max(1, _env_int(env, "SCHEDULEURM_PROBE_ROUTE_MAX_WORKERS", "4")),
        "NODE_PROBE_CACHE_TTL_S": max(0, _env_int(env, "SCHEDULEURM_NODE_PROBE_CACHE_TTL_S", "300")),
        "DISPATCH_PROBE_CACHE_MAX_AGE_S": max(
            0, _env_int(env, "SCHEDULEURM_DISPATCH_PROBE_CACHE_MAX_AGE_S", "30")
        ),
        "WATCH_PROBE_CACHE_MAX_AGE_S": max(
            0, _env_int(env, "SCHEDULEURM_WATCH_PROBE_CACHE_MAX_AGE_S", "30")
        ),
        # Cache-derived cycle snapshots are refreshed immediately before
        # dispatch at this age. A snapshot collected live in the same watcher
        # tick is reused regardless, so slow terminal checks cannot trigger a
        # second consecutive multi-node SSH sweep.
        "WATCH_PRE_DISPATCH_PROBE_MAX_AGE_S": max(
            0,
            _env_int(
                env,
                "SCHEDULEURM_WATCH_PRE_DISPATCH_PROBE_MAX_AGE_S",
                "20",
            ),
        ),
        "DISK_FULL_PATTERNS": (
            "No space left on device",
            "[Errno 28]",
            "ENOSPC",
            "disk full",
            "Disk quota exceeded",
        ),
        "INVALID_FLAG_PATTERNS": (
            "FATAL Flags parsing error",
            "Unknown command line flag",
            "argparse.ArgumentError",
            "error: unrecognized arguments",
            "error: argument",
            "error: the following arguments are required",
            "Error: Got unexpected extra argument",
            "Error: No such option",
            "Error: Missing option",
            "No such command",
        ),
        "OOM_PATTERNS": (
            "CUDA out of memory",
            "out of memory",
            "MemoryError",
            "Killed process",
            "oom-kill",
            "oom_reaper",
        ),
        "SSH_ROUTE_CACHE_TTL_S": _env_int(env, "SCHEDULEURM_SSH_ROUTE_CACHE_TTL_S", "300"),
        "SSH_LOGIN_PREFLIGHT_TIMEOUT_S": max(1, _env_int(env, "SCHEDULEURM_SSH_LOGIN_PREFLIGHT_TIMEOUT_S", "15")),
        "SSH_LOGIN_FAILURE_CACHE_TTL_S": max(0, _env_int(env, "SCHEDULEURM_SSH_LOGIN_FAILURE_CACHE_TTL_S", "60")),
        "SSH_DISABLE_MUX": _env_bool(env, "SCHEDULEURM_SSH_DISABLE_MUX", "1"),
        "_SSH_PROXY_JUMP_CACHE": {},
        "_SSH_ROUTE_JUMP_CACHE": {},
        "_SSH_OUTER_ROUTE_CACHE": {},
        "_RELAY_NODE_CACHE": {},
        "STARTUP_FLOOR_MB": 500,
        "EVICT_TASK_MIN_AGE_S": 180,
        "MIGRATION_LOAD_RATIO": _env_float(env, "SCHEDULEURM_MIGRATION_LOAD_RATIO", "2.0"),
        "MIGRATION_FREE_THRESHOLD_S": _env_int(env, "SCHEDULEURM_MIGRATION_FREE_THRESHOLD_S", "600"),
        "MIGRATION_MAX_PER_DISPATCH": _env_int(env, "SCHEDULEURM_MIGRATION_MAX_PER_DISPATCH", "1"),
        "MIGRATION_MIN_TASK_ETA_S": migration_min_task_eta_s,
        "MIGRATION_COOLDOWN_S": _env_int(env, "SCHEDULEURM_MIGRATION_COOLDOWN_S", "1800"),
        "MIGRATION_MIN_SOURCE_LOAD_S": _env_int(env, "SCHEDULEURM_MIGRATION_MIN_SOURCE_LOAD_S", "600"),
        "MIGRATION_MAX_CWD_SIZE_MB": _env_int(env, "SCHEDULEURM_MIGRATION_MAX_CWD_SIZE_MB", "1024"),
        "LAUNCH_MAX_CWD_SIZE_MB": _env_int(env, "SCHEDULEURM_LAUNCH_MAX_CWD_SIZE_MB", "2048"),
        "LAUNCH_STAGING_MAX_CANDIDATES_PER_PASS": _env_int(
            env, "SCHEDULEURM_LAUNCH_STAGING_MAX_CANDIDATES_PER_PASS", "256"
        ),
        "LAUNCH_STAGING_WORKERS": max(
            1, _env_int(env, "SCHEDULEURM_LAUNCH_STAGING_WORKERS", "12")
        ),
        "MIGRATION_MAX_CKPT_SIZE_MB": _env_int(env, "SCHEDULEURM_MIGRATION_MAX_CKPT_SIZE_MB", "2048"),
        "RESUME_CKPT_MIGRATE_MIN_WAIT_S": _env_int(
            env, "SCHEDULEURM_RESUME_CKPT_MIGRATE_MIN_WAIT_S", str(migration_min_task_eta_s)
        ),
        "RESUME_CKPT_MIGRATE_EST_MBPS": max(1.0, _env_float(env, "SCHEDULEURM_RESUME_CKPT_MIGRATE_EST_MBPS", "40")),
        "RESUME_CKPT_MIGRATE_OVERHEAD_S": _env_int(env, "SCHEDULEURM_RESUME_CKPT_MIGRATE_OVERHEAD_S", "60"),
        "RESUME_CKPT_MIGRATE_SAFETY_S": _env_int(env, "SCHEDULEURM_RESUME_CKPT_MIGRATE_SAFETY_S", "120"),
        "STAGING_FAIL_COOLDOWN_S": _env_int(env, "SCHEDULEURM_STAGING_FAIL_COOLDOWN_S", "3600"),
        "STAGING_TTL_S": _env_int(env, "SCHEDULEURM_STAGING_TTL_S", "600"),
        "RESULT_SYNC_MAX_ATTEMPTS": _env_int(env, "SCHEDULEURM_RESULT_SYNC_MAX_ATTEMPTS", "5"),
        "RESULT_SYNC_TIMEOUT_S": result_sync_timeout_s,
        "RESULT_SYNC_STALE_GRACE_S": _env_int(env, "SCHEDULEURM_RESULT_SYNC_STALE_GRACE_S", "600"),
        "RESULT_SYNC_MAX_PER_CYCLE": max(1, _env_int(env, "SCHEDULEURM_RESULT_SYNC_MAX_PER_CYCLE", "1")),
        "RESULT_SYNC_DEPENDENCY_MAX_PER_CYCLE": max(
            1,
            _env_int(env, "SCHEDULEURM_RESULT_SYNC_DEPENDENCY_MAX_PER_CYCLE", "8"),
        ),
        "EVICT_RELAUNCH_COOLDOWN_S": _env_int(env, "SCHED_EVICT_RELAUNCH_COOLDOWN_S", str(30 * 60)),
        "LOCAL_CPU_EVICT_NODE_COOLDOWN_S": _env_int(env, "SCHED_LOCAL_CPU_EVICT_NODE_COOLDOWN_S", str(30 * 60)),
        "LOCAL_GPU_HOST_CPU_BLOCK_PCT": _env_int(env, "SCHED_LOCAL_GPU_HOST_CPU_BLOCK_PCT", "85"),
        "LOCAL_GPU_HOST_CPU_EVICT_PCT": _env_int(env, "SCHED_LOCAL_GPU_HOST_CPU_EVICT_PCT", "92"),
        "PREEMPT_QUEUE_WAIT_MIN": 5,
        "PREEMPT_VICTIM_MIN_AGE_MIN": 10,
        "PREEMPT_VICTIM_MAX_AGE_MIN": 240,
        "PREEMPT_MAX_VICTIMS_PER_DISPATCH": 3,
        "REBOOT_DETECT_WINDOW_S": 600,
    }
