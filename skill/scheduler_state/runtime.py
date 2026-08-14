"""Runtime-bound state, lock, dispatch-intent, and cache compatibility exports."""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional

from scheduler_dispatch.intent import (
    clear_dispatch_intent as _clear_dispatch_intent_file,
    dispatch_intent_message as _dispatch_intent_message_impl,
    read_dispatch_intent as _read_dispatch_intent_file,
    write_dispatch_intent as _write_dispatch_intent_file,
)
from scheduler_state.locks import (
    SchedulerLockTimeout,
    counting_file_lock as _counting_file_lock_impl,
    launch_exec_lock as _launch_exec_lock_impl,
    log_lock_delay as _log_lock_delay_impl,
    lock_timeout_value as _lock_timeout_value_impl,
    state_lock as _state_lock_impl,
    watcher_lifetime_lock as _watcher_lifetime_lock_impl,
)
from scheduler_probe.cache import (
    load_node_probe_cache as _load_node_probe_cache_impl,
    save_node_probe_cache as _save_node_probe_cache_impl,
)
from scheduler_state.io import (
    atomic_write_json as _atomic_write_json_impl,
    empty_queue_reset_allowed as _empty_queue_reset_allowed_impl,
    guard_empty_queue_save as _guard_empty_queue_save_impl,
    guard_missing_queue_state as _guard_missing_queue_state_impl,
    load_json as _load_json_impl,
    load_scheduler_state as _load_scheduler_state_impl,
    recent_queue_state_files as _recent_queue_state_files_impl,
    save_scheduler_state as _save_scheduler_state_impl,
    state_counts as _state_counts_impl,
)


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def _pid_alive_local(pid) -> bool:
    try:
        pid = int(pid)
    except Exception:
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def build_state_runtime_exports(namespace: Mapping[str, Any]) -> dict[str, Any]:
    def _log_state_lock_delay(purpose: str, shared: bool, wait_s: float, hold_s: float):
        return _log_lock_delay_impl(
            log_dir=_ns(namespace, "LOG_DIR"),
            purpose=purpose,
            shared=shared,
            wait_s=wait_s,
            hold_s=hold_s,
            warn_wait_s=_ns(namespace, "STATE_LOCK_WARN_WAIT_S"),
            warn_hold_s=_ns(namespace, "STATE_LOCK_WARN_HOLD_S"),
            argv=sys.argv[:],
        )

    def _lock_timeout_value(timeout_s):
        return _lock_timeout_value_impl(timeout_s)

    def state_lock(timeout_s=None, shared=False, purpose="state"):
        return _state_lock_impl(
            lock_file=_ns(namespace, "LOCK_FILE"),
            log_dir=_ns(namespace, "LOG_DIR"),
            timeout_s=timeout_s,
            shared=shared,
            purpose=purpose,
            warn_wait_s=_ns(namespace, "STATE_LOCK_WARN_WAIT_S"),
            warn_hold_s=_ns(namespace, "STATE_LOCK_WARN_HOLD_S"),
            argv=sys.argv[:],
        )

    def history_lock(timeout_s=None, purpose="history"):
        return _state_lock_impl(
            lock_file=_ns(namespace, "HISTORY_LOCK_FILE"),
            log_dir=_ns(namespace, "LOG_DIR"),
            timeout_s=timeout_s,
            shared=False,
            purpose=purpose,
            warn_wait_s=_ns(namespace, "STATE_LOCK_WARN_WAIT_S"),
            warn_hold_s=_ns(namespace, "STATE_LOCK_WARN_HOLD_S"),
            argv=sys.argv[:],
        )

    def result_sync_worker_lock(timeout_s=0, purpose="result-sync:worker"):
        return _state_lock_impl(
            lock_file=_ns(namespace, "RESULT_SYNC_WORKER_LOCK_FILE"),
            log_dir=_ns(namespace, "LOG_DIR"),
            timeout_s=timeout_s,
            shared=False,
            purpose=purpose,
            warn_wait_s=_ns(namespace, "STATE_LOCK_WARN_WAIT_S"),
            warn_hold_s=_ns(namespace, "STATE_LOCK_WARN_HOLD_S"),
            argv=sys.argv[:],
        )

    def launch_exec_lock(timeout_s=None, purpose="launch-exec"):
        return _launch_exec_lock_impl(
            lock_file=_ns(namespace, "LAUNCH_EXEC_LOCK_FILE"),
            log_dir=_ns(namespace, "LOG_DIR"),
            default_timeout_s=_ns(namespace, "LAUNCH_EXEC_LOCK_TIMEOUT_S"),
            timeout_s=timeout_s,
            purpose=purpose,
            warn_wait_s=_ns(namespace, "STATE_LOCK_WARN_WAIT_S"),
            warn_hold_s=_ns(namespace, "STATE_LOCK_WARN_HOLD_S"),
            argv=sys.argv[:],
        )

    def _launch_exec_slot_prefix(node: str) -> Path:
        safe_node = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(node or "unknown")).strip("_")
        if not safe_node:
            safe_node = "unknown"
        return _ns(namespace, "LAUNCH_EXEC_LOCK_FILE").with_name(
            f"{_ns(namespace, 'LAUNCH_EXEC_LOCK_FILE').name}.{safe_node}"
        )

    def launch_exec_slot_lock(node: str, timeout_s=None, purpose="launch-exec"):
        timeout = _ns(namespace, "LAUNCH_EXEC_LOCK_TIMEOUT_S") if timeout_s is None else timeout_s
        node_label = str(node or "unknown")
        return _counting_file_lock_impl(
            lock_file_prefix=_launch_exec_slot_prefix(node_label),
            slots=_ns(namespace, "LAUNCH_EXEC_MAX_PER_NODE"),
            log_dir=_ns(namespace, "LOG_DIR"),
            timeout_s=timeout,
            purpose=f"{purpose}:{node_label}",
            timeout_label="scheduler launch slot lock",
            warn_wait_s=_ns(namespace, "STATE_LOCK_WARN_WAIT_S"),
            warn_hold_s=_ns(namespace, "STATE_LOCK_WARN_HOLD_S"),
            argv=sys.argv[:],
        )

    def watcher_lifetime_lock(timeout_s=0, purpose="watch:lifetime"):
        return _watcher_lifetime_lock_impl(
            lock_file=_ns(namespace, "WATCHER_LOCK_FILE"),
            log_dir=_ns(namespace, "LOG_DIR"),
            timeout_s=timeout_s,
            purpose=purpose,
            argv=sys.argv[:],
        )

    def _read_dispatch_intent() -> Optional[dict]:
        return _read_dispatch_intent_file(
            _ns(namespace, "DISPATCH_INTENT_FILE"),
            pid_alive=_pid_alive_local,
        )

    def _write_dispatch_intent(label: str = "", ttl_s: Optional[float] = None):
        return _write_dispatch_intent_file(
            _ns(namespace, "DISPATCH_INTENT_FILE"),
            label=label,
            ttl_s=ttl_s,
            default_ttl_s=_ns(namespace, "DISPATCH_INTENT_TTL_S"),
            pid=os.getpid(),
        )

    def _clear_dispatch_intent():
        _clear_dispatch_intent_file(_ns(namespace, "DISPATCH_INTENT_FILE"), pid=os.getpid())

    def _dispatch_intent_message(intent: dict) -> str:
        return _dispatch_intent_message_impl(intent)

    def _load_json(path, default):
        return _load_json_impl(path, default, now_fn=time.time)

    def _atomic_write_json(path, obj):
        return _atomic_write_json_impl(path, obj)

    def _empty_queue_reset_allowed() -> bool:
        return _empty_queue_reset_allowed_impl(_ns(namespace, "ALLOW_EMPTY_QUEUE_RESET_ENV"))

    def _state_counts(obj) -> tuple[int, int]:
        return _state_counts_impl(obj)

    def _recent_queue_state_files() -> list[Path]:
        return _recent_queue_state_files_impl(
            _ns(namespace, "STATE_DIR"),
            max_age_s=_ns(namespace, "QUEUE_MISSING_GUARD_MAX_AGE_S"),
            now_fn=time.time,
        )

    def _guard_missing_queue_state():
        return _guard_missing_queue_state_impl(
            queue_file=_ns(namespace, "QUEUE_FILE"),
            state_dir=_ns(namespace, "STATE_DIR"),
            allow_env=_ns(namespace, "ALLOW_EMPTY_QUEUE_RESET_ENV"),
            max_age_s=_ns(namespace, "QUEUE_MISSING_GUARD_MAX_AGE_S"),
            now_fn=time.time,
        )

    def _guard_empty_queue_save(new_state):
        return _guard_empty_queue_save_impl(
            new_state,
            queue_file=_ns(namespace, "QUEUE_FILE"),
            allow_env=_ns(namespace, "ALLOW_EMPTY_QUEUE_RESET_ENV"),
            min_tasks=_ns(namespace, "EMPTY_QUEUE_RESET_GUARD_MIN_TASKS"),
            min_active=_ns(namespace, "EMPTY_QUEUE_RESET_GUARD_MIN_ACTIVE"),
        )

    def load_state():
        return _load_scheduler_state_impl(
            queue_file=_ns(namespace, "QUEUE_FILE"),
            state_dir=_ns(namespace, "STATE_DIR"),
            allow_env=_ns(namespace, "ALLOW_EMPTY_QUEUE_RESET_ENV"),
            missing_guard_max_age_s=_ns(namespace, "QUEUE_MISSING_GUARD_MAX_AGE_S"),
            canonicalize=_ns(namespace, "_canonicalize_state_node_names"),
        )

    def save_state(state):
        return _save_scheduler_state_impl(
            state,
            queue_file=_ns(namespace, "QUEUE_FILE"),
            allow_env=_ns(namespace, "ALLOW_EMPTY_QUEUE_RESET_ENV"),
            empty_guard_min_tasks=_ns(namespace, "EMPTY_QUEUE_RESET_GUARD_MIN_TASKS"),
            empty_guard_min_active=_ns(namespace, "EMPTY_QUEUE_RESET_GUARD_MIN_ACTIVE"),
            canonicalize=_ns(namespace, "_canonicalize_state_node_names"),
            compact=_ns(namespace, "_compact_state_for_persistence"),
        )

    def load_history():
        return _load_json(_ns(namespace, "VRAM_FILE"), {})

    def save_history(history):
        return _atomic_write_json(_ns(namespace, "VRAM_FILE"), history)

    def load_runtime_history():
        return _load_json(_ns(namespace, "RUNTIME_FILE"), {})

    def save_runtime_history(history):
        return _atomic_write_json(_ns(namespace, "RUNTIME_FILE"), history)

    def _save_node_probe_cache(states: dict) -> None:
        return _save_node_probe_cache_impl(
            _ns(namespace, "NODE_PROBE_CACHE_FILE"),
            states,
            now_fn=time.time,
        )

    def _load_node_probe_cache(max_age_s: Optional[int] = None) -> dict:
        return _load_node_probe_cache_impl(
            _ns(namespace, "NODE_PROBE_CACHE_FILE"),
            default_ttl_s=_ns(namespace, "NODE_PROBE_CACHE_TTL_S"),
            max_age_s=max_age_s,
            now_fn=time.time,
        )

    return {
        "SchedulerLockTimeout": SchedulerLockTimeout,
        "_pid_alive_local": _pid_alive_local,
        "_log_state_lock_delay": _log_state_lock_delay,
        "_lock_timeout_value": _lock_timeout_value,
        "state_lock": state_lock,
        "history_lock": history_lock,
        "result_sync_worker_lock": result_sync_worker_lock,
        "launch_exec_lock": launch_exec_lock,
        "_launch_exec_slot_prefix": _launch_exec_slot_prefix,
        "launch_exec_slot_lock": launch_exec_slot_lock,
        "watcher_lifetime_lock": watcher_lifetime_lock,
        "_read_dispatch_intent": _read_dispatch_intent,
        "_write_dispatch_intent": _write_dispatch_intent,
        "_clear_dispatch_intent": _clear_dispatch_intent,
        "_dispatch_intent_message": _dispatch_intent_message,
        "_load_json": _load_json,
        "_atomic_write_json": _atomic_write_json,
        "_empty_queue_reset_allowed": _empty_queue_reset_allowed,
        "_state_counts": _state_counts,
        "_recent_queue_state_files": _recent_queue_state_files,
        "_guard_missing_queue_state": _guard_missing_queue_state,
        "_guard_empty_queue_save": _guard_empty_queue_save,
        "load_state": load_state,
        "save_state": save_state,
        "load_history": load_history,
        "save_history": save_history,
        "load_runtime_history": load_runtime_history,
        "save_runtime_history": save_runtime_history,
        "_save_node_probe_cache": _save_node_probe_cache,
        "_load_node_probe_cache": _load_node_probe_cache,
    }
