"""Outside-lock migration candidate staging runner."""

from __future__ import annotations

from .types import MigrationCandidateStagingDeps


def stage_migration_candidates_outside_lock(
    *,
    max_candidates: int = 2,
    nodes: list | None = None,
    deps: MigrationCandidateStagingDeps,
) -> None:
    """Identify migration candidates under a short lock, then stage them outside the lock."""
    try:
        with deps.state_lock(shared=True, purpose="migration-staging:snapshot"):
            state = deps.load_state()
        if nodes is None:
            may_migrate = any(
                task.get("status") == "queued"
                and task.get("preferred_node")
                and not task.get("require_node")
                and not task.get("auto_adopted")
                and int(task.get("eta_seconds") or 0) > 0
                for task in state.get("tasks", [])
            )
            if not may_migrate:
                return
            nodes = deps.probe_all()
        snapshot = deps.identify_migration_candidates(
            state,
            nodes,
            max_candidates=max_candidates,
        )
    except Exception as exc:
        try:
            deps.notify(
                "migration_snapshot_error",
                {"error": str(exc)[:200]},
                feishu_enabled=False,
            )
        except Exception:
            pass
        return

    if not snapshot:
        return

    if len(deps.staged_tasks) >= deps.staged_tasks_max:
        sorted_keys = sorted(deps.staged_tasks.items(), key=lambda kv: kv[1])
        for key, _value in sorted_keys[:deps.staged_tasks_max // 4]:
            deps.staged_tasks.pop(key, None)

    for task_snapshot, target in snapshot:
        cache_key = (task_snapshot["id"], target)
        if cache_key in deps.staged_tasks:
            continue
        try:
            ok, msg = deps.stage_for_migration(task_snapshot, target)
            if ok:
                deps.staged_tasks[cache_key] = deps.now()
                continue
            deps.record_staging_failure(task_snapshot["id"], target)
            try:
                deps.notify(
                    "migration_staging_skip",
                    {
                        "task_id": task_snapshot["id"],
                        "target": target,
                        "reason": msg[:200],
                    },
                    feishu_enabled=False,
                )
            except Exception:
                pass
        except Exception as exc:
            deps.record_staging_failure(task_snapshot["id"], target)
            try:
                deps.notify(
                    "migration_staging_error",
                    {
                        "task_id": task_snapshot["id"],
                        "target": target,
                        "error": str(exc)[:200],
                    },
                    feishu_enabled=False,
                )
            except Exception:
                pass
