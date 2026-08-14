"""Plan launch cwd/checkpoint staging candidates outside the state lock."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

try:
    from scheduler_staging.launch_cwd import unsafe_staging_cwd_reason
except ModuleNotFoundError:
    from ..scheduler_staging.launch_cwd import unsafe_staging_cwd_reason


@dataclass(frozen=True)
class LaunchStagingPlanDeps:
    node_configs: dict
    queued_wait_for_file_block_reason: Callable[[dict], str]
    hpc_cpu_pool_soft_require_nodes: Callable[[dict], list]
    task_requires_resume_scan: Callable[[dict], bool]
    node_is_windows: Callable[[str], bool]
    node_cpu_fallback_block_reason: Callable[[dict, str, dict], str]
    staging_cache_hit: Callable[[tuple], bool]
    staging_cap_recent: Callable[[tuple], bool]
    resume_checkpoint_stage_check: Callable[[dict, str, list], tuple[str, dict | None, str]]


@dataclass(frozen=True)
class LaunchStagingPlan:
    protected_under_cwd: dict[str, set[str]] = field(default_factory=dict)
    candidates: set[tuple[str, str]] = field(default_factory=set)
    ckpt_candidates: list[tuple[dict, str, dict]] = field(default_factory=list)
    input_candidates: list[tuple[dict, str]] = field(default_factory=list)
    gpu_staging_pending: bool = False


def build_protected_under_cwd(tasks: list[dict]) -> dict[str, set[str]]:
    """Return cwd-relative paths that rsync --delete must never remove."""
    protected: dict[str, set[str]] = {}
    for task in tasks:
        cwd = task.get("cwd")
        if not cwd:
            continue
        if unsafe_staging_cwd_reason(cwd):
            continue
        cwd_norm = str(cwd).rstrip("/")
        for exclude in (task.get("stage_excludes") or []):
            exclude = str(exclude or "").strip().strip("/")
            if exclude:
                protected.setdefault(cwd_norm, set()).add(exclude)
        for field in ("ckpt_dir", "result_dir", "local_result_dir"):
            path = task.get(field)
            if not path:
                continue
            path_norm = str(path).rstrip("/")
            try:
                if (
                    path_norm == cwd_norm
                    or os.path.commonpath([path_norm, cwd_norm]) == cwd_norm
                ):
                    rel = os.path.relpath(path_norm, cwd_norm)
                    if rel and rel != "." and not rel.startswith(".."):
                        protected.setdefault(cwd_norm, set()).add(rel.rstrip("/") + "/")
            except (ValueError, OSError):
                pass
    return protected


def _queued_tasks_for_staging(
    tasks: list[dict],
    task_ids: set[str],
    deps: LaunchStagingPlanDeps,
) -> list[dict]:
    queued = [task for task in tasks if task.get("status") == "queued"]
    if task_ids:
        queued = [task for task in queued if str(task.get("id") or "") in task_ids]
    high_gpu_tasks = [
        task for task in queued
        if task.get("priority") == "high" and int(task.get("est_vram_mb") or 0) > 0
    ]
    if not high_gpu_tasks:
        return queued
    remaining_tasks = [
        task for task in queued
        if not (
            task.get("priority") == "high"
            and int(task.get("est_vram_mb") or 0) > 0
        )
    ]
    # Priority controls staging order, not eligibility. A blocked high-priority
    # GPU task must not remove unrelated normal GPU work from every staging
    # pass, or its sticky launch-input target can remain deferred forever.
    return high_gpu_tasks + remaining_tasks


def _target_nodes_for_task(task: dict, deps: LaunchStagingPlanDeps) -> list[str]:
    require = task.get("require_node")
    soft_require_pool = deps.hpc_cpu_pool_soft_require_nodes(task)
    if soft_require_pool:
        targets = list(soft_require_pool)
    elif require:
        targets = [require]
    else:
        targets = list(deps.node_configs.keys())
    allowed = set(task.get("allowed_nodes") or [])
    if allowed:
        targets = [target for target in targets if target in allowed]
    if (
        not require
        and deps.task_requires_resume_scan(task)
        and task.get("resume_locations")
        and "local" in deps.node_configs
        and "local" not in targets
    ):
        targets.append("local")
    return targets


def _target_is_eligible_for_task(
    task: dict,
    target: str,
    *,
    require: str | None,
    soft_require_pool: list,
    allowed: set,
    gpu_staging_pending: bool,
    skip_launch_staging: bool,
    deps: LaunchStagingPlanDeps,
) -> bool:
    node_info = deps.node_configs.get(target, {})
    if node_info.get("host") is None:
        return False
    if skip_launch_staging and node_info.get("skip_launch_staging"):
        return False
    task_needs_gpu = int(task.get("est_vram_mb") or 0) > 0
    if gpu_staging_pending and task_needs_gpu and deps.node_is_windows(target):
        return False
    if node_info.get("stage_only_when_targeted"):
        targeted = (
            require == target
            or task.get("preferred_node") == target
            or target in allowed
            or target in soft_require_pool
        )
        if not targeted:
            return False
    cpu_fallback_target = (
        task_needs_gpu
        and (deps.node_is_windows(target) or node_info.get("max_vram_per_task") == 0)
    )
    if cpu_fallback_target and deps.node_cpu_fallback_block_reason(task, target, node_info):
        return False
    return True


def _gpu_staging_pending(tasks: list[dict], deps: LaunchStagingPlanDeps) -> bool:
    return any(
        int(task.get("est_vram_mb") or 0) > 0
        and not deps.queued_wait_for_file_block_reason(task)
        for task in tasks
    )


def _launch_input_candidate_identity(
    task: dict, target: str, node_configs: dict,
) -> tuple:
    """Cheap per-pass identity; content hashing remains in the staging backend."""
    explicit = [
        os.path.realpath(os.path.expanduser(str(path)))
        for path in (task.get("stage_input_paths") or [])
        if str(path).strip()
    ]
    if explicit:
        paths = explicit
    else:
        cwd = os.path.realpath(os.path.expanduser(str(task.get("cwd") or "")))
        paths = []
        for raw_path in task.get("wait_for_files") or []:
            parent = os.path.dirname(
                os.path.realpath(os.path.expanduser(str(raw_path)))
            )
            try:
                if parent == cwd or os.path.commonpath([parent, cwd]) != cwd:
                    continue
            except ValueError:
                continue
            paths.append(parent)
    target_info = node_configs.get(target, {}) or {}
    target_scope = str(
        target_info.get("shared_workspace_group") or target)
    return target_scope, tuple(sorted(dict.fromkeys(paths)))


def collect_launch_staging_plan(
    state: dict,
    task_ids: set[str] | None,
    *,
    deps: LaunchStagingPlanDeps,
) -> LaunchStagingPlan:
    tasks = list(state.get("tasks", []))
    target_ids = {str(task_id) for task_id in (task_ids or set()) if str(task_id)}
    queued_tasks = _queued_tasks_for_staging(tasks, target_ids, deps)
    gpu_staging_pending = _gpu_staging_pending(queued_tasks, deps)
    candidates: set[tuple[str, str]] = set()
    ckpt_candidates: list[tuple[dict, str, dict]] = []
    input_candidates: dict[tuple, tuple[dict, str]] = {}
    planned_ckpt_counts: dict[str, int] = {}

    for task in queued_tasks:
        waiting_for_files = bool(deps.queued_wait_for_file_block_reason(task))
        cwd = task.get("cwd")
        if not cwd:
            continue
        require = task.get("require_node")
        soft_require_pool = deps.hpc_cpu_pool_soft_require_nodes(task)
        allowed = set(task.get("allowed_nodes") or [])
        targets = _target_nodes_for_task(task, deps)
        input_target = str(task.get("stage_input_target") or "")
        stage_options: list[tuple[str, dict]] = []
        if not task.get("skip_launch_staging"):
            for target in targets:
                if not _target_is_eligible_for_task(
                    task,
                    target,
                    require=require,
                    soft_require_pool=soft_require_pool,
                    allowed=allowed,
                    gpu_staging_pending=gpu_staging_pending,
                    skip_launch_staging=True,
                    deps=deps,
                ):
                    continue
                cwd_key = ("local", target, cwd)
                if deps.staging_cache_hit(cwd_key) or deps.staging_cap_recent(cwd_key):
                    continue
                candidates.add((target, cwd))

        # A prerequisite blocks launch and checkpoint placement, not a source
        # refresh.  Stage corrected code while an upstream task is producing
        # its artifacts so the dependent task cannot run a stale module once
        # its file gate opens.
        if waiting_for_files:
            continue
        if input_target:
            if input_target in targets:
                input_key = _launch_input_candidate_identity(
                    task, input_target, deps.node_configs
                )
                input_candidates.setdefault(
                    input_key, (dict(task), input_target))
        if not deps.task_requires_resume_scan(task):
            continue
        locations = list(task.get("resume_locations") or [])
        if not locations:
            continue
        # Dispatch keeps a launch-input target sticky across cycles. Resume
        # staging must follow the same target; otherwise each watcher pass can
        # copy a large checkpoint to another eligible node while the task is
        # still waiting for its declared inputs.
        checkpoint_targets = (
            [input_target]
            if input_target and input_target in targets
            else targets
        )
        for target in checkpoint_targets:
            if not _target_is_eligible_for_task(
                task,
                target,
                require=require,
                soft_require_pool=soft_require_pool,
                allowed=allowed,
                gpu_staging_pending=gpu_staging_pending,
                skip_launch_staging=False,
                deps=deps,
            ):
                continue
            state_ckpt, source_loc, _ = deps.resume_checkpoint_stage_check(
                task, target, locations)
            if state_ckpt != "needs_stage" or not source_loc:
                continue
            stage_options.append((target, dict(source_loc)))

        if stage_options:
            # A checkpoint is task-specific and can be hundreds of MB.  Stage
            # it to one candidate per pass instead of fanning it out to every
            # eligible node.  Balance tasks across targets so a bulk dispatch
            # can still fill the cluster in one pass.
            target_order = {
                target: index for index, target in enumerate(targets)}
            target, source_loc = min(
                stage_options,
                key=lambda item: (
                    planned_ckpt_counts.get(item[0], 0),
                    target_order[item[0]],
                ),
            )
            ckpt_candidates.append((dict(task), target, source_loc))
            planned_ckpt_counts[target] = (
                planned_ckpt_counts.get(target, 0) + 1)

    return LaunchStagingPlan(
        protected_under_cwd=build_protected_under_cwd(tasks),
        candidates=candidates,
        ckpt_candidates=ckpt_candidates,
        input_candidates=list(input_candidates.values()),
        gpu_staging_pending=gpu_staging_pending,
    )


def launch_stage_candidate_key(
    item: tuple[str, str],
    *,
    node_configs: dict,
    node_is_windows: Callable[[str], bool],
) -> tuple:
    target, cwd = item
    node_info = node_configs.get(target, {})
    caps = set(str(cap).lower() for cap in (node_info.get("capabilities") or []))
    if node_is_windows(target):
        node_rank = 3
    elif "cuda" in caps or "jax_cuda" in caps or "torch_cuda" in caps:
        node_rank = 0
    else:
        node_rank = 1
    return (node_rank, target, cwd)
