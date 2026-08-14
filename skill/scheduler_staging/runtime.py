"""Runtime-bound staging, migration-cache, and conda-sync compatibility exports."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from scheduler_staging.coordination import (
    mark_persistent_staging_success,
    persistent_staging_marker_hit,
    staging_key_digest,
    staging_key_guard,
)

from scheduler_staging.state import (
    can_migrate_to as _can_migrate_to_impl,
    conda_sync_ok as _conda_sync_ok_impl,
    mark_launch_stage_cap_exceeded as _mark_launch_stage_cap_exceeded_impl,
    mark_launch_stage_success as _mark_launch_stage_success_impl,
    mark_staging_cache_success as _mark_staging_cache_success_impl,
    recently_failed as _staging_recently_failed_impl,
    record_conda_sync_failed as _record_conda_sync_failed_impl,
    record_conda_sync_ok as _record_conda_sync_ok_impl,
    record_staging_failure as _record_staging_failure_impl,
    stage_cwd_check as _stage_cwd_check_impl,
    stage_failure_reason as _stage_failure_reason_impl,
    ttl_cache_hit as _staging_cache_hit_impl,
)


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_staging_runtime_exports(namespace: Mapping[str, Any]) -> dict[str, Any]:
    # Timestamped rsync success cache. Key: (source_node, target_node, path).
    _STAGING_CACHE: dict = {}

    # Migration candidate failures, keyed by (task_id, target_node).
    _STAGING_FAILED: dict = {}
    _STAGING_FAILED_MAX = 200

    # Conda environment preload success markers, keyed by (node, env_path).
    _CONDA_SYNC_OK: dict = {}

    # Launch-time cwd/checkpoint staging state.
    _STAGING_CAP_EXCEEDED: dict = {}
    _STAGING_FAILS: dict = {}

    # Migration staging completion markers, keyed by (task_id, target_node).
    _STAGED_TASKS: dict = {}
    _STAGED_TASKS_MAX = 100
    _STAGING_RESOLUTION_STAMPS: dict[tuple, tuple[int, int]] = {}

    def _now() -> float:
        return _ns(namespace, "time").time()

    def _coord_root():
        return _ns(namespace, "STATE_DIR") / "staging_coord"

    def _shared_workspace_keys(key: tuple) -> list[tuple]:
        if len(key) == 3 and key[0] == "local":
            target_index = 1
        elif len(key) == 4 and key[0] == "resume_ckpt":
            target_index = 2
        else:
            return [key]
        target = key[target_index]
        group = (_ns(namespace, "NODES").get(target, {}) or {}).get(
            "shared_workspace_group"
        )
        if not group:
            return [key]
        peers = sorted(
            name
            for name, info in _ns(namespace, "NODES").items()
            if (info or {}).get("shared_workspace_group") == group
            and not (info or {}).get("retired")
        )
        equivalents = []
        for peer in peers:
            candidate = list(key)
            candidate[target_index] = peer
            equivalents.append(tuple(candidate))
        return equivalents or [key]

    def _canonical_coord_key(key: tuple) -> tuple:
        return _shared_workspace_keys(key)[0]

    def _cache_key(key: tuple) -> tuple:
        """Invalidate staging markers when configured source files change."""
        if len(key) != 3 or key[0] != "local":
            return key
        info = (_ns(namespace, "NODES").get(key[1], {}) or {})
        method = str(info.get("launch_staging_method") or "")
        include_paths = tuple(str(item) for item in (
            info.get("launch_stage_include_paths") or []))
        if not include_paths:
            return key
        exclude_paths = tuple(
            str(item).replace("\\", "/").strip("/")
            for item in (
                info.get("launch_stage_fingerprint_exclude_paths") or []
            )
            if str(item).strip("/\\")
        )

        def excluded_from_fingerprint(relative: Path) -> bool:
            relative_text = relative.as_posix().strip("/")
            return any(
                relative_text == excluded
                or relative_text.startswith(excluded + "/")
                for excluded in exclude_paths
            )

        # A marker is only valid for the staging topology that produced it.
        # In particular, moving a node from a shared workspace to a private
        # workspace must force a fresh copy even when source files are unchanged.
        workspace_scope = str(info.get("shared_workspace_group") or f"node:{key[1]}")
        source_snapshot = []
        root = Path(key[2])
        for rel in include_paths:
            path = root / rel
            candidates = [path] if path.is_file() else (
                sorted(path.rglob("*")) if path.is_dir() else [])
            for candidate in candidates:
                if not candidate.is_file():
                    continue
                relative = candidate.relative_to(root)
                if (
                    "__pycache__" in relative.parts
                    or ".pytest_cache" in relative.parts
                    or candidate.suffix == ".pyc"
                    or excluded_from_fingerprint(relative)
                ):
                    continue
                try:
                    stat = candidate.stat()
                except OSError:
                    continue
                source_snapshot.append((
                    str(relative), int(stat.st_mtime_ns), int(stat.st_size)))
        payload = (
            method,
            include_paths,
            exclude_paths,
            workspace_scope,
            tuple(source_snapshot),
        )
        return (*key, f"source:{staging_key_digest(payload)}")

    def _resolve_staged_escalations(keys: list[tuple]) -> None:
        resolver = namespace.get("_resolve_staged_cwd_escalations")
        if resolver is None or not keys or keys[0][0] != "local":
            return
        cache_key = (keys[0][2], tuple(key[1] for key in keys))
        escalation_file = namespace.get("ESCALATIONS_FILE")
        try:
            stat = escalation_file.stat() if escalation_file is not None else None
            stamp = (int(stat.st_mtime_ns), int(stat.st_size)) if stat is not None else (0, 0)
        except FileNotFoundError:
            stamp = (0, 0)
        if _STAGING_RESOLUTION_STAMPS.get(cache_key) == stamp:
            return
        try:
            resolver(keys[0][2], [key[1] for key in keys])
        except Exception:
            return
        try:
            stat = escalation_file.stat() if escalation_file is not None else None
            stamp = (int(stat.st_mtime_ns), int(stat.st_size)) if stat is not None else (0, 0)
        except FileNotFoundError:
            stamp = (0, 0)
        _STAGING_RESOLUTION_STAMPS[cache_key] = stamp

    def _staging_cache_hit(key) -> bool:
        equivalent_keys = _shared_workspace_keys(key)
        for candidate in equivalent_keys:
            candidate_cache_key = _cache_key(candidate)
            if _staging_cache_hit_impl(
                _STAGING_CACHE,
                candidate_cache_key,
                ttl_s=_ns(namespace, "STAGING_TTL_S"),
                now=_now,
            ):
                stamp = _now()
                for equivalent in equivalent_keys:
                    _STAGING_CACHE[_cache_key(equivalent)] = stamp
                _resolve_staged_escalations(equivalent_keys)
                return True
            if persistent_staging_marker_hit(
                _coord_root(),
                candidate_cache_key,
                ttl_s=_ns(namespace, "STAGING_TTL_S"),
                now=_now,
            ):
                stamp = _now()
                for equivalent in equivalent_keys:
                    _STAGING_CACHE[_cache_key(equivalent)] = stamp
                _resolve_staged_escalations(equivalent_keys)
                return True
        return False

    def _staging_key_guard(key):
        return staging_key_guard(
            _coord_root(), _cache_key(_canonical_coord_key(key)))

    def _staging_recently_failed(task_id, target) -> bool:
        return _staging_recently_failed_impl(
            _STAGING_FAILED,
            task_id,
            target,
            cooldown_s=_ns(namespace, "STAGING_FAIL_COOLDOWN_S"),
            now=_now,
        )

    def _record_staging_failure(task_id, target):
        return _record_staging_failure_impl(
            _STAGING_FAILED,
            task_id,
            target,
            max_entries=_STAGING_FAILED_MAX,
            now=_now,
        )

    def _record_conda_sync_ok(node, env_path):
        return _record_conda_sync_ok_impl(_CONDA_SYNC_OK, node, env_path, now=_now)

    def _record_conda_sync_failed(node, env_path):
        return _record_conda_sync_failed_impl(_CONDA_SYNC_OK, node, env_path)

    def _conda_sync_ok(node, env_path) -> bool:
        return _conda_sync_ok_impl(
            _CONDA_SYNC_OK,
            node,
            env_path,
            ttl_s=_ns(namespace, "STAGING_TTL_S"),
            now=_now,
        )

    def _stage_cwd_check(target_node: str, cwd: str):
        return _stage_cwd_check_impl(
            target_node,
            cwd,
            node_configs=_ns(namespace, "NODES"),
            staging_cache_hit=_staging_cache_hit,
            staging_cap_exceeded=_STAGING_CAP_EXCEEDED,
            staging_fails=_STAGING_FAILS,
            staging_ttl_s=_ns(namespace, "STAGING_TTL_S"),
            staging_fail_cooldown_s=_ns(namespace, "STAGING_FAIL_COOLDOWN_S"),
            now=_now,
        )

    def _stage_failure_reason(target_node: str, cwd: str) -> str:
        return _stage_failure_reason_impl(_STAGING_FAILS, target_node, cwd)

    def _mark_launch_stage_success(key: tuple) -> None:
        equivalent_keys = _shared_workspace_keys(key)
        for equivalent in equivalent_keys:
            cache_key = _cache_key(equivalent)
            _mark_launch_stage_success_impl(
                _STAGING_CACHE,
                _STAGING_CAP_EXCEEDED,
                _STAGING_FAILS,
                cache_key,
                now=_now,
            )
            if cache_key != equivalent:
                _STAGING_CAP_EXCEEDED.pop(equivalent, None)
                _STAGING_FAILS.pop(equivalent, None)
            mark_persistent_staging_success(
                _coord_root(), cache_key, now=_now)
        _resolve_staged_escalations(equivalent_keys)

    def _mark_launch_stage_cap_exceeded(key: tuple) -> None:
        for equivalent in _shared_workspace_keys(key):
            _mark_launch_stage_cap_exceeded_impl(
                _STAGING_CAP_EXCEEDED,
                equivalent,
                now=_now,
            )

    def _mark_staging_cache_success(key: tuple) -> None:
        for equivalent in _shared_workspace_keys(key):
            cache_key = _cache_key(equivalent)
            _mark_staging_cache_success_impl(
                _STAGING_CACHE, cache_key, now=_now
            )
            mark_persistent_staging_success(
                _coord_root(), cache_key, now=_now
            )

    def _can_migrate_to(task: dict, target_node: str, timeout_s: int = 5) -> bool:
        del timeout_s
        return _can_migrate_to_impl(
            task,
            target_node,
            staged_tasks=_STAGED_TASKS,
            staging_ttl_s=_ns(namespace, "STAGING_TTL_S"),
            now=_now,
        )

    return {
        "_STAGING_CACHE": _STAGING_CACHE,
        "_STAGING_FAILED": _STAGING_FAILED,
        "_STAGING_FAILED_MAX": _STAGING_FAILED_MAX,
        "_CONDA_SYNC_OK": _CONDA_SYNC_OK,
        "_STAGING_CAP_EXCEEDED": _STAGING_CAP_EXCEEDED,
        "_STAGING_FAILS": _STAGING_FAILS,
        "_STAGED_TASKS": _STAGED_TASKS,
        "_STAGED_TASKS_MAX": _STAGED_TASKS_MAX,
        "_STAGING_RESOLUTION_STAMPS": _STAGING_RESOLUTION_STAMPS,
        "_staging_cache_hit": _staging_cache_hit,
        "_staging_key_guard": _staging_key_guard,
        "_staging_recently_failed": _staging_recently_failed,
        "_record_staging_failure": _record_staging_failure,
        "_record_conda_sync_ok": _record_conda_sync_ok,
        "_record_conda_sync_failed": _record_conda_sync_failed,
        "_conda_sync_ok": _conda_sync_ok,
        "_stage_cwd_check": _stage_cwd_check,
        "_stage_failure_reason": _stage_failure_reason,
        "_mark_launch_stage_success": _mark_launch_stage_success,
        "_mark_launch_stage_cap_exceeded": _mark_launch_stage_cap_exceeded,
        "_mark_staging_cache_success": _mark_staging_cache_success,
        "_can_migrate_to": _can_migrate_to,
    }
