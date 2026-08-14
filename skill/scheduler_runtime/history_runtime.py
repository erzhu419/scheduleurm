"""Runtime-bound task runtime-history and ETA seed wrappers for scheduler.py."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Mapping, Optional


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def load_runtime_history_cached(path, loader, cache: dict) -> dict:
    """Reuse a read-only history snapshot until the atomic file changes."""
    try:
        stat = Path(path).stat()
        signature = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
    except OSError:
        return loader()
    if cache.get("signature") == signature and "value" in cache:
        return cache["value"]
    value = loader()
    try:
        stat = Path(path).stat()
        signature = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
    except OSError:
        return value
    cache["signature"] = signature
    cache["value"] = value
    return value


def build_runtime_history_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    candidate_history_cache: dict[str, Any] = {}
    candidate_closest_index_cache: dict[str, Any] = {}

    def _candidate_runtime_history() -> dict:
        return load_runtime_history_cached(
            _ns(namespace, "RUNTIME_FILE"),
            _ns(namespace, "load_runtime_history"),
            candidate_history_cache,
        )

    def _candidate_runtime_closest_index(history: Optional[dict] = None) -> list[dict]:
        history = _candidate_runtime_history() if history is None else history
        if candidate_closest_index_cache.get("history") is history:
            return candidate_closest_index_cache["value"]
        value = _ns(namespace, "_runtime_history_closest_index_impl")(history)
        candidate_closest_index_cache["history"] = history
        candidate_closest_index_cache["value"] = value
        return value

    def _task_runtime_payload(task: dict) -> dict:
        return _ns(namespace, "_task_runtime_payload_impl")(
            task,
            conflict_path_key=_ns(namespace, "_conflict_path_key"),
        )

    def _task_runtime_keys(task: dict) -> list:
        return _ns(namespace, "_task_runtime_keys_impl")(
            task,
            task_runtime_payload=_ns(namespace, "_task_runtime_payload"),
        )

    def _runtime_device_kind_for_task(task: dict, gpu_idx=None) -> str:
        return _ns(namespace, "_runtime_device_kind_for_task_impl")(
            task,
            gpu_idx,
            task_launch_cpu_mode=_ns(namespace, "_task_launch_cpu_mode"),
        )

    def _runtime_node_bucket_key(node: str, device_kind: str) -> str:
        return _ns(namespace, "_runtime_node_bucket_key_impl")(
            node,
            device_kind,
            canonical_node_name=_ns(namespace, "_canonical_node_name"),
        )

    def _runtime_node_bucket_keys(node: str, device_kind: str) -> list[str]:
        return _ns(namespace, "_runtime_node_bucket_keys_impl")(
            node,
            device_kind,
            canonical_node_name=_ns(namespace, "_canonical_node_name"),
            node_name_aliases=_ns(namespace, "NODE_NAME_ALIASES"),
        )

    def _candidate_runtime_seconds(task: dict, node: str, gpu_idx=None) -> int:
        return _ns(namespace, "_candidate_runtime_seconds_impl")(
            task,
            node,
            gpu_idx,
            load_runtime_history=_candidate_runtime_history,
            task_runtime_keys=_ns(namespace, "_task_runtime_keys"),
            runtime_device_kind_for_task=_ns(namespace, "_runtime_device_kind_for_task"),
            runtime_node_bucket_keys=_ns(namespace, "_runtime_node_bucket_keys"),
        )

    def _load_eta_tracker_module():
        scheduler_file = namespace.get("__file__") or __file__
        return _ns(namespace, "_load_eta_tracker_module_impl")(Path(scheduler_file).parent)

    def _runtime_total_units_from_cmd(cmd: str) -> int:
        return _ns(namespace, "_runtime_total_units_from_cmd_impl")(
            cmd,
            load_eta_tracker_module_fn=_ns(namespace, "_load_eta_tracker_module"),
        )

    def _runtime_history_lookup_deps():
        return _ns(namespace, "_build_runtime_history_lookup_deps")(namespace)

    def _runtime_history_closest(
        task: dict,
        history: dict,
        closest_index: Optional[list[dict]] = None,
    ):
        return _ns(namespace, "_runtime_history_closest_impl")(
            task,
            history,
            task_runtime_payload=_ns(namespace, "_task_runtime_payload"),
            deps=_ns(namespace, "_runtime_history_lookup_deps")(),
            closest_index=closest_index,
        )

    def _runtime_history_best(
        task: dict,
        runtime_history: Optional[dict] = None,
        closest_index: Optional[list[dict]] = None,
    ):
        return _ns(namespace, "_runtime_history_best_impl")(
            task,
            task_runtime_payload=_ns(namespace, "_task_runtime_payload"),
            deps=_ns(namespace, "_runtime_history_lookup_deps")(),
            runtime_history=runtime_history,
            closest_index=closest_index,
        )

    def _runtime_total_history_s(
        task: dict,
        runtime_history: Optional[dict] = None,
        closest_index: Optional[list[dict]] = None,
    ) -> int:
        return _ns(namespace, "_runtime_total_history_s_impl")(
            task,
            task_runtime_payload=_ns(namespace, "_task_runtime_payload"),
            deps=_ns(namespace, "_runtime_history_lookup_deps")(),
            runtime_history=runtime_history,
            closest_index=closest_index,
        )

    def _history_eta_for_task(
        task: dict,
        runtime_history: Optional[dict] = None,
        closest_index: Optional[list[dict]] = None,
        resource_history: Optional[dict] = None,
    ) -> tuple[int, str]:
        return _ns(namespace, "_history_eta_for_task_impl")(
            task,
            task_runtime_payload=_ns(namespace, "_task_runtime_payload"),
            deps=_ns(namespace, "_runtime_history_lookup_deps")(),
            runtime_history=runtime_history,
            closest_index=closest_index,
            resource_history=resource_history,
        )

    def _seed_pending_eta_from_history(
        state: dict,
        runtime_history_cache: Optional[dict] = None,
        runtime_closest_index: Optional[list[dict]] = None,
        resource_history_cache: Optional[dict] = None,
    ) -> int:
        return _ns(namespace, "_seed_pending_eta_from_history_impl")(
            state,
            task_runtime_payload=_ns(namespace, "_task_runtime_payload"),
            deps=_ns(namespace, "_runtime_history_lookup_deps")(),
            runtime_history_cache=runtime_history_cache,
            runtime_closest_index=runtime_closest_index,
            resource_history_cache=resource_history_cache,
        )

    def _runtime_walltime_for_task(task: dict) -> int:
        return _ns(namespace, "_runtime_walltime_for_task_impl")(
            task,
            task_runtime_payload=_ns(namespace, "_task_runtime_payload"),
            deps=_ns(namespace, "_runtime_history_lookup_deps")(),
        )

    def _apply_runtime_projection(task: dict, projection: Optional[dict]):
        return _ns(namespace, "_apply_runtime_projection_impl")(task, projection, now=time.time)

    def _runtime_history_record_deps():
        return _ns(namespace, "_build_runtime_history_record_deps")(namespace)

    def runtime_history_record(task: dict, duration_s: int = 0):
        history_batch = _ns(namespace, "_history_batch_coordinator")
        if history_batch.queue_runtime(task, duration_s):
            return None
        with _ns(namespace, "history_lock")(purpose="history:runtime-record"):
            return _ns(namespace, "_runtime_history_record_impl")(
                task,
                duration_s=duration_s,
                deps=_ns(namespace, "_runtime_history_record_deps")(),
            )

    def _runtime_profile_from_log(
        log_path: str,
        cmd: str = "",
        observed_duration_s: float = 0,
    ):
        return _ns(namespace, "_runtime_profile_from_log_impl")(
            log_path,
            cmd=cmd,
            observed_duration_s=observed_duration_s,
            load_eta_tracker_module_fn=_ns(namespace, "_load_eta_tracker_module"),
            read_text_tail_fn=_ns(namespace, "_read_text_tail"),
        )

    def _apply_test_log_runtime_profile(task: dict, test_log: str) -> bool:
        return _ns(namespace, "_apply_test_log_runtime_profile_impl")(
            task,
            test_log,
            deps=_ns(namespace, "_RuntimeProfileDeps")(
                runtime_profile_from_log=_ns(namespace, "_runtime_profile_from_log"),
                apply_runtime_projection=_ns(namespace, "_apply_runtime_projection"),
                runtime_history_record=_ns(namespace, "runtime_history_record"),
                expanduser=os.path.expanduser,
            ),
        )

    return {
        "_candidate_runtime_history": _candidate_runtime_history,
        "_candidate_runtime_closest_index": _candidate_runtime_closest_index,
        "_task_runtime_payload": _task_runtime_payload,
        "_task_runtime_keys": _task_runtime_keys,
        "_runtime_device_kind_for_task": _runtime_device_kind_for_task,
        "_runtime_node_bucket_key": _runtime_node_bucket_key,
        "_runtime_node_bucket_keys": _runtime_node_bucket_keys,
        "_candidate_runtime_seconds": _candidate_runtime_seconds,
        "_load_eta_tracker_module": _load_eta_tracker_module,
        "_runtime_total_units_from_cmd": _runtime_total_units_from_cmd,
        "_runtime_history_lookup_deps": _runtime_history_lookup_deps,
        "_runtime_history_closest": _runtime_history_closest,
        "_runtime_history_best": _runtime_history_best,
        "_runtime_total_history_s": _runtime_total_history_s,
        "_history_eta_for_task": _history_eta_for_task,
        "_seed_pending_eta_from_history": _seed_pending_eta_from_history,
        "_runtime_walltime_for_task": _runtime_walltime_for_task,
        "_apply_runtime_projection": _apply_runtime_projection,
        "_runtime_history_record_deps": _runtime_history_record_deps,
        "runtime_history_record": runtime_history_record,
        "_runtime_profile_from_log": _runtime_profile_from_log,
        "_apply_test_log_runtime_profile": _apply_test_log_runtime_profile,
    }
