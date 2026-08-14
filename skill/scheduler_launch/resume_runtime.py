"""Runtime-bound resume scan constants and wrappers."""

from __future__ import annotations

import os
import re
import uuid
from typing import Any, Mapping, Optional

try:
    from scheduler_migration.resume_scan import (
        apply_resume_scan_result as _apply_resume_scan_result_impl,
        allow_initial_resume_scan_error as _allow_initial_resume_scan_error_impl,
        cached_resume_locations_for_task as _cached_resume_locations_for_task_impl,
        find_resume_on_node as _find_resume_on_node_impl,
        refresh_resume_locations_for_task as _refresh_resume_locations_for_task_impl,
        resume_candidate_is_safe as _resume_candidate_is_safe_impl,
        resume_node_names as _resume_node_names_impl,
        resume_scan_key as _resume_scan_key_impl,
        scan_resume_locations as _scan_resume_locations_impl,
        task_requires_resume_scan as _task_requires_resume_scan_impl,
    )
    from scheduler_migration.resume_prefetch import (
        ResumePrefetchDeps as _ResumePrefetchDeps,
        prefetch_resume_scans_outside_lock as _prefetch_resume_scans_outside_lock_impl,
    )
    from scheduler_runtime.wiring import build_resume_scan_deps as _build_resume_scan_deps
except ModuleNotFoundError:
    from ..scheduler_migration.resume_scan import (
        apply_resume_scan_result as _apply_resume_scan_result_impl,
        allow_initial_resume_scan_error as _allow_initial_resume_scan_error_impl,
        cached_resume_locations_for_task as _cached_resume_locations_for_task_impl,
        find_resume_on_node as _find_resume_on_node_impl,
        refresh_resume_locations_for_task as _refresh_resume_locations_for_task_impl,
        resume_candidate_is_safe as _resume_candidate_is_safe_impl,
        resume_node_names as _resume_node_names_impl,
        resume_scan_key as _resume_scan_key_impl,
        scan_resume_locations as _scan_resume_locations_impl,
        task_requires_resume_scan as _task_requires_resume_scan_impl,
    )
    from ..scheduler_migration.resume_prefetch import (
        ResumePrefetchDeps as _ResumePrefetchDeps,
        prefetch_resume_scans_outside_lock as _prefetch_resume_scans_outside_lock_impl,
    )
    from ..scheduler_runtime_wiring import build_resume_scan_deps as _build_resume_scan_deps


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_launch_resume_runtime_exports(namespace: Mapping[str, Any]) -> dict[str, Any]:
    exports: dict[str, Any] = {
        "CKPT_EXTS": (
            "pt",
            "pth",
            "pkl",
            "ckpt",
            "bin",
            "safetensors",
            "npy",
            "npz",
            "h5",
            "hdf5",
            "tar",
        ),
        "RESUME_SAFE_NAME_RE": re.compile(
            r"(checkpoint|ckpt|resume|state|snapshot|epoch|step|iter|iteration)",
            re.IGNORECASE,
        ),
        "RESUME_UNSAFE_NAME_RE": re.compile(
            r"(^|[_\-.])(model_?final|final_?model|final|best|buffer|replay|rollout|metrics?|results?|eval|train_?log)([_\-.]|$)",
            re.IGNORECASE,
        ),
        "RESUME_SCAN_TTL_S": int(os.environ.get("SCHEDULEURM_RESUME_SCAN_TTL_S", "900")),
        "RESUME_SCAN_WORKERS": max(
            1,
            int(os.environ.get("SCHEDULEURM_RESUME_SCAN_WORKERS", "4")),
        ),
        "RESUME_PREFETCH_MAX_TASKS_PER_PASS": max(
            1,
            int(os.environ.get("SCHEDULEURM_RESUME_PREFETCH_MAX_TASKS_PER_PASS", "48")),
        ),
        "RESUME_PREFETCH_TASK_WORKERS": max(
            1,
            int(os.environ.get("SCHEDULEURM_RESUME_PREFETCH_TASK_WORKERS", "4")),
        ),
        "RESUME_SCAN_INFLIGHT_TTL_S": max(
            60,
            int(os.environ.get("SCHEDULEURM_RESUME_SCAN_INFLIGHT_TTL_S", "600")),
        ),
    }

    def _resume_scan_deps():
        return _build_resume_scan_deps(namespace)

    def _resume_candidate_is_safe(path: str, explicit_glob: bool = False) -> bool:
        return _resume_candidate_is_safe_impl(
            path,
            explicit_glob=explicit_glob,
            deps=_ns(namespace, "_resume_scan_deps")(),
        )

    def _resume_node_names(task: Optional[dict] = None, nodes: Optional[list] = None) -> list:
        return _resume_node_names_impl(
            task,
            nodes,
            deps=_ns(namespace, "_resume_scan_deps")(),
        )

    def _resume_scan_key(task: dict, nodes: Optional[list] = None) -> list:
        return _resume_scan_key_impl(
            task,
            nodes,
            deps=_ns(namespace, "_resume_scan_deps")(),
        )

    def _task_requires_resume_scan(task: dict) -> bool:
        return _task_requires_resume_scan_impl(
            task,
            deps=_ns(namespace, "_resume_scan_deps")(),
        )

    def _allow_initial_resume_scan_error(task: dict) -> bool:
        return _allow_initial_resume_scan_error_impl(task)

    def _find_resume_on_node(task: dict, node: str) -> Optional[dict]:
        return _find_resume_on_node_impl(
            task,
            node,
            deps=_ns(namespace, "_resume_scan_deps")(),
        )

    def scan_resume_locations(
        task: dict,
        nodes: Optional[list] = None,
        cache: Optional[dict] = None,
    ) -> tuple:
        return _scan_resume_locations_impl(
            task,
            nodes=nodes,
            cache=cache,
            deps=_ns(namespace, "_resume_scan_deps")(),
        )

    def _refresh_resume_locations_for_task(task: dict, nodes: list, cache: dict) -> tuple:
        return _refresh_resume_locations_for_task_impl(
            task,
            nodes,
            cache,
            deps=_ns(namespace, "_resume_scan_deps")(),
        )

    def _cached_resume_locations_for_task(task: dict, nodes: list) -> tuple:
        return _cached_resume_locations_for_task_impl(
            task,
            nodes,
            deps=_ns(namespace, "_resume_scan_deps")(),
        )

    def _apply_resume_scan_result(task: dict, **kwargs) -> None:
        return _apply_resume_scan_result_impl(task, **kwargs)

    def _prefetch_resume_scans_outside_lock(
        nodes: list,
        task_ids: Optional[set[str]] = None,
    ) -> dict:
        return _prefetch_resume_scans_outside_lock_impl(
            nodes,
            task_ids,
            deps=_ResumePrefetchDeps(
                state_lock=_ns(namespace, "state_lock"),
                load_state=_ns(namespace, "load_state"),
                save_state=_ns(namespace, "save_state"),
                task_requires_resume_scan=_ns(namespace, "_task_requires_resume_scan"),
                resume_scan_key=_ns(namespace, "_resume_scan_key"),
                cached_resume_locations_for_task=_cached_resume_locations_for_task,
                scan_resume_locations=scan_resume_locations,
                apply_resume_scan_result=_apply_resume_scan_result,
                notify=_ns(namespace, "notify"),
                max_tasks_per_pass=_ns(namespace, "RESUME_PREFETCH_MAX_TASKS_PER_PASS"),
                task_workers=_ns(namespace, "RESUME_PREFETCH_TASK_WORKERS"),
                inflight_ttl_s=_ns(namespace, "RESUME_SCAN_INFLIGHT_TTL_S"),
                now=_ns(namespace, "time").time,
                token_factory=lambda task: (
                    f"{os.getpid()}:{task.get('id')}:{uuid.uuid4().hex}"
                ),
            ),
        )

    def _resume_location_for_node(task: dict, node: str) -> Optional[dict]:
        for loc in task.get("resume_locations") or []:
            if loc.get("node") == node:
                return loc
        return None

    def find_resume(task):
        node = task.get("node")
        if not node:
            return None
        try:
            found = _ns(namespace, "_find_resume_on_node")(task, node)
        except Exception:
            return None
        return (found or {}).get("path") or None

    exports.update({
        "_resume_scan_deps": _resume_scan_deps,
        "_resume_candidate_is_safe": _resume_candidate_is_safe,
        "_resume_node_names": _resume_node_names,
        "_resume_scan_key": _resume_scan_key,
        "_task_requires_resume_scan": _task_requires_resume_scan,
        "_allow_initial_resume_scan_error": _allow_initial_resume_scan_error,
        "_find_resume_on_node": _find_resume_on_node,
        "scan_resume_locations": scan_resume_locations,
        "_refresh_resume_locations_for_task": _refresh_resume_locations_for_task,
        "_cached_resume_locations_for_task": _cached_resume_locations_for_task,
        "_apply_resume_scan_result": _apply_resume_scan_result,
        "_prefetch_resume_scans_outside_lock": _prefetch_resume_scans_outside_lock,
        "_resume_location_for_node": _resume_location_for_node,
        "find_resume": find_resume,
    })
    return exports
