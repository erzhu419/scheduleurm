"""Shared dataclasses and low-level helpers for migration staging."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable

try:
    from scheduler_data_motion.rsync import run_local_rsync
except ModuleNotFoundError:
    from ..scheduler_data_motion.rsync import run_local_rsync


@dataclass(frozen=True)
class MigrationStagingDeps:
    node_configs: dict
    migration_max_cwd_size_mb: int
    staging_cache_hit: Callable[[tuple], bool]
    mark_staging_success: Callable[[tuple], None]
    remote_path_for_node: Callable[[str, str], str]
    rsync_path_for_node: Callable[[str, str], str]
    rsync_shell_for_pair: Callable[[str, str], str | None]
    run_on: Callable[..., tuple[int, str, str]]
    apply_node_cmd_rewrites: Callable[[str, str], str]
    run_subprocess: Callable[..., Any] = subprocess.run


@dataclass(frozen=True)
class MigrationCandidateStagingDeps:
    state_lock: Callable[..., Any]
    load_state: Callable[[], dict]
    probe_all: Callable[[], list]
    identify_migration_candidates: Callable[..., list]
    stage_for_migration: Callable[[dict, str], tuple[bool, str]]
    record_staging_failure: Callable[[str, str], Any]
    notify: Callable[..., Any]
    staged_tasks: dict
    staged_tasks_max: int
    now: Callable[[], float] = time.time


def node_host(deps: MigrationStagingDeps, node: str | None) -> str | None:
    return (deps.node_configs.get(node or "", {}) or {}).get("host")


def run_migration_rsync(args: list[str], *, kind: str, deps: MigrationStagingDeps) -> tuple[bool, str]:
    return run_local_rsync(
        args,
        operation=f"rsync {kind}",
        run_subprocess=deps.run_subprocess,
        timeout_s=600,
        timeout_human=">10min",
        failure_word=" failed",
        exception_limit=10_000,
    )
