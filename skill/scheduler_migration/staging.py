"""Migration cwd/checkpoint/env staging facade.

The implementation is packaged under scheduler_migration/; this module keeps the
legacy import surface used by scheduler.py and older tests.
"""

from __future__ import annotations

from .candidate_staging import stage_migration_candidates_outside_lock
from .ckpt_staging import (
    probe_ckpt_size as _probe_ckpt_size,
    source_ckpt_presence as _source_ckpt_presence,
    stage_checkpoint_if_needed as _stage_checkpoint_if_needed,
    stage_ckpt as _stage_ckpt,
)
from .cwd_staging import stage_cwd as _stage_cwd
from .env_staging import verify_python_env as _verify_python_env
from .types import (
    MigrationCandidateStagingDeps,
    MigrationStagingDeps,
    node_host as _node_host,
    run_migration_rsync as _run_rsync,
)


def stage_for_migration(
    task: dict,
    target_node: str,
    *,
    max_ckpt_mb: int,
    deps: MigrationStagingDeps,
) -> tuple[bool, str]:
    """Stage cwd, checkpoint, and python env prerequisites before migration."""
    cwd = task.get("cwd")
    if not cwd:
        return False, "no cwd on task"

    source_node = task.get("preferred_node") or task.get("node")
    if source_node and source_node == target_node:
        return True, "source==target; nothing to stage"

    ok, msg = _stage_cwd(source_node, target_node, cwd, deps=deps)
    if not ok:
        return False, msg

    ok, msg, ckpt_staged = _stage_checkpoint_if_needed(
        task,
        source_node,
        target_node,
        max_ckpt_mb=max_ckpt_mb,
        deps=deps,
    )
    if not ok:
        return False, msg

    ok, msg, env_checked = _verify_python_env(task, target_node, deps=deps)
    if not ok:
        return False, msg

    return True, f"staged (cwd{' + ckpt' if ckpt_staged else ''}{' + env' if env_checked else ''})"


__all__ = [
    "MigrationCandidateStagingDeps",
    "MigrationStagingDeps",
    "_node_host",
    "_probe_ckpt_size",
    "_run_rsync",
    "_source_ckpt_presence",
    "_stage_checkpoint_if_needed",
    "_stage_ckpt",
    "_stage_cwd",
    "_verify_python_env",
    "stage_for_migration",
    "stage_migration_candidates_outside_lock",
]
