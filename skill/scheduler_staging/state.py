"""Shared TTL and marker state helpers for staging workflows."""

from __future__ import annotations

from typing import Callable, Mapping

try:
    from scheduler_staging.launch_cwd import unsafe_staging_cwd_reason
except ModuleNotFoundError:
    from .launch_cwd import unsafe_staging_cwd_reason


def ttl_cache_hit(cache: dict, key, *, ttl_s: int, now: Callable[[], float]) -> bool:
    ts = cache.get(key)
    if ts is None:
        return False
    if (now() - ts) > ttl_s:
        cache.pop(key, None)
        return False
    return True


def recently_failed(
    failed: dict,
    task_id,
    target,
    *,
    cooldown_s: int,
    now: Callable[[], float],
) -> bool:
    key = (task_id, target)
    ts = failed.get(key)
    if ts is None:
        return False
    if (now() - ts) > cooldown_s:
        failed.pop(key, None)
        return False
    return True


def record_staging_failure(
    failed: dict,
    task_id,
    target,
    *,
    max_entries: int,
    now: Callable[[], float],
) -> None:
    if len(failed) >= max_entries:
        sorted_keys = sorted(failed.items(), key=lambda kv: kv[1])
        for k, _ in sorted_keys[:max(1, max_entries // 4)]:
            failed.pop(k, None)
    failed[(task_id, target)] = now()


def record_conda_sync_ok(conda_sync_ok: dict, node, env_path, *, now: Callable[[], float]) -> None:
    conda_sync_ok[(node, env_path)] = now()


def record_conda_sync_failed(conda_sync_ok: dict, node, env_path) -> None:
    conda_sync_ok.pop((node, env_path), None)


def conda_sync_ok(
    conda_sync_ok: dict,
    node,
    env_path,
    *,
    ttl_s: int,
    now: Callable[[], float],
) -> bool:
    ts = conda_sync_ok.get((node, env_path))
    if ts is None:
        return False
    if (now() - ts) > ttl_s:
        conda_sync_ok.pop((node, env_path), None)
        return False
    return True


def stage_cwd_check(
    target_node: str,
    cwd: str,
    *,
    node_configs: Mapping,
    staging_cache_hit: Callable[[tuple], bool],
    staging_cap_exceeded: dict,
    staging_fails: dict,
    staging_ttl_s: int,
    staging_fail_cooldown_s: int,
    now: Callable[[], float],
) -> str:
    if node_configs.get(target_node, {}).get("host") is None:
        return "ready"
    if node_configs.get(target_node, {}).get("skip_launch_staging"):
        return "ready"
    if unsafe_staging_cwd_reason(cwd):
        return "unsafe_path"
    cwd_key = ("local", target_node, cwd)
    if staging_cache_hit(cwd_key):
        return "ready"
    ts = staging_cap_exceeded.get(cwd_key)
    if ts is not None:
        if (now() - ts) > staging_ttl_s:
            staging_cap_exceeded.pop(cwd_key, None)
        else:
            return "cap_exceeded"
    fail = staging_fails.get(cwd_key)
    if fail is not None:
        fail_ts = fail[0] if isinstance(fail, tuple) else fail
        if (now() - fail_ts) > staging_fail_cooldown_s:
            staging_fails.pop(cwd_key, None)
        else:
            return "stage_failed"
    return "needs_stage"


def stage_failure_reason(staging_fails: dict, target_node: str, cwd: str) -> str:
    cwd_key = ("local", target_node, cwd)
    fail = staging_fails.get(cwd_key)
    if fail is None:
        return ""
    return fail[1] if isinstance(fail, tuple) and len(fail) > 1 else "unknown"


def mark_launch_stage_success(
    staging_cache: dict,
    staging_cap_exceeded: dict,
    staging_fails: dict,
    key: tuple,
    *,
    now: Callable[[], float],
) -> None:
    staging_cache[key] = now()
    staging_cap_exceeded.pop(key, None)
    staging_fails.pop(key, None)


def mark_launch_stage_cap_exceeded(
    staging_cap_exceeded: dict,
    key: tuple,
    *,
    now: Callable[[], float],
) -> None:
    staging_cap_exceeded[key] = now()


def mark_staging_cache_success(staging_cache: dict, key: tuple, *, now: Callable[[], float]) -> None:
    staging_cache[key] = now()


def can_migrate_to(
    task: dict,
    target_node: str,
    *,
    staged_tasks: dict,
    staging_ttl_s: int,
    now: Callable[[], float],
) -> bool:
    key = (task.get("id"), target_node)
    ts = staged_tasks.get(key)
    if ts is None:
        return False
    if (now() - ts) > staging_ttl_s:
        staged_tasks.pop(key, None)
        return False
    return True
