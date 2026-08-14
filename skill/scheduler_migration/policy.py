"""Load-balance migration policy helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


def compute_node_load_seconds(state: dict, *, node_configs: dict) -> dict:
    """Sum ETA seconds for active or pinned future work per known node."""
    loads: dict = {name: 0 for name in node_configs}
    for task in state.get("tasks", []):
        if task.get("auto_adopted"):
            continue
        eta = int(task.get("eta_seconds") or 0)
        if eta <= 0:
            continue
        status = task.get("status")
        if status in ("running", "launching"):
            node = task.get("node")
            if node in loads:
                loads[node] += eta
        elif status == "queued":
            pin = task.get("require_node") or task.get("preferred_node")
            if pin and pin in loads:
                loads[pin] += eta
    return loads


@dataclass(frozen=True)
class MigrationPolicyDeps:
    compute_node_load_seconds: Callable[[dict], dict]
    blocked_nodes_for_task: Callable[[dict], set]
    launch_failed_nodes_for_task: Callable[[dict], set]
    staging_recently_failed: Callable[[str, str], bool]
    can_migrate_to: Callable[[dict, str], bool]
    migration_free_threshold_s: int
    migration_load_ratio: float
    migration_min_source_load_s: int
    migration_min_task_eta_s: int
    migration_cooldown_s: int
    migration_max_per_dispatch: int
    now: Callable[[], float]


def _alive_load_extremes(loads: dict, nodes: list, deps: MigrationPolicyDeps):
    if not loads or len(loads) < 2:
        return None
    alive_names = {
        node["name"]
        for node in nodes
        if node.get("alive") and not node.get("measurement_reservation_active")
    }
    candidate_loads = [(name, load) for name, load in loads.items() if name in alive_names]
    if len(candidate_loads) < 2:
        return None
    candidate_loads.sort(key=lambda item: item[1])
    target_name, target_load = candidate_loads[0]
    source_name, source_load = candidate_loads[-1]
    if source_name == target_name:
        return None
    if target_load >= deps.migration_free_threshold_s:
        return None
    if source_load < deps.migration_load_ratio * max(target_load, 1):
        return None
    if source_load < deps.migration_min_source_load_s:
        return None
    return source_name, source_load, target_name, target_load


def _migration_candidate_eta(task: dict, deps: MigrationPolicyDeps) -> int | None:
    eta = int(task.get("eta_seconds") or 0)
    if eta < deps.migration_min_task_eta_s:
        return None
    return eta


def _migration_cooldown_active(task: dict, deps: MigrationPolicyDeps) -> bool:
    last_mig_at = float(task.get("migrated_at") or 0)
    return bool(last_mig_at and (deps.now() - last_mig_at) < deps.migration_cooldown_s)


def _base_candidate_allowed(task: dict, source_name: str, target_name: str, deps: MigrationPolicyDeps) -> bool:
    if task.get("status") != "queued":
        return False
    if task.get("require_node"):
        return False
    if task.get("auto_adopted"):
        return False
    if task.get("preferred_node") != source_name:
        return False
    if target_name in deps.blocked_nodes_for_task(task):
        return False
    if target_name in deps.launch_failed_nodes_for_task(task):
        return False
    if _migration_cooldown_active(task, deps):
        return False
    return True


def identify_migration_candidates(
    state: dict,
    nodes: list,
    max_candidates: int = 2,
    *,
    deps: MigrationPolicyDeps,
) -> list:
    """Return up to max_candidates (task_snapshot, target_node) pairs without I/O."""
    loads = deps.compute_node_load_seconds(state)
    extremes = _alive_load_extremes(loads, nodes, deps)
    if extremes is None:
        return []
    source_name, _source_load, target_name, _target_load = extremes

    candidates = []
    for task in state.get("tasks", []):
        if not _base_candidate_allowed(task, source_name, target_name, deps):
            continue
        eta = _migration_candidate_eta(task, deps)
        if eta is None:
            continue
        task_id = task.get("id")
        if deps.staging_recently_failed(task_id, target_name):
            continue
        candidates.append({
            "id": task_id,
            "cwd": task.get("cwd"),
            "ckpt_dir": task.get("ckpt_dir"),
            "cmd": task.get("cmd"),
            "preferred_node": task.get("preferred_node"),
            "signature": task.get("signature"),
            "eta_seconds": eta,
        })
    candidates.sort(key=lambda task: int(task.get("eta_seconds") or 0))
    return [(candidate, target_name) for candidate in candidates[:max_candidates]]


def consider_migration(
    state: dict,
    nodes: list,
    loads: Optional[dict] = None,
    *,
    deps: MigrationPolicyDeps,
) -> list:
    """Commit migration policy decisions by mutating eligible queued tasks in state."""
    if loads is None:
        loads = deps.compute_node_load_seconds(state)
    extremes = _alive_load_extremes(loads, nodes, deps)
    if extremes is None:
        return []
    source_name, source_load, target_name, target_load = extremes

    candidates = []
    for task in state.get("tasks", []):
        if not _base_candidate_allowed(task, source_name, target_name, deps):
            continue
        eta = _migration_candidate_eta(task, deps)
        if eta is None:
            continue
        candidates.append(task)
    if not candidates:
        return []

    candidates.sort(key=lambda task: int(task.get("eta_seconds") or 0))
    migrated = []
    for candidate in candidates:
        if len(migrated) >= deps.migration_max_per_dispatch:
            break
        if not deps.can_migrate_to(candidate, target_name):
            continue
        old_pref = candidate.get("preferred_node")
        candidate["preferred_node"] = target_name
        candidate["staged_node"] = target_name
        candidate["migrated_from"] = old_pref
        candidate["migrated_at"] = deps.now()
        candidate["last_block_reason"] = (
            f"migrated: preferred {old_pref}(load={int(source_load)}s) \u2192 "
            f"{target_name}(load={int(target_load)}s) for load balance"
        )
        migrated.append(candidate["id"])
    return migrated
