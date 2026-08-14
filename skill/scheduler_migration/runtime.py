"""Runtime-bound resume checkpoint and migration wrappers for scheduler.py."""

from __future__ import annotations

from typing import Any, Mapping, Optional


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_migration_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    def _migration_staging_deps():
        return _ns(namespace, "_build_migration_staging_deps")(namespace)

    def _resume_ckpt_stage_key(
        source_node: str,
        target_node: str,
        ckpt_dir: str,
        source_location: Optional[dict] = None,
    ) -> tuple:
        return _ns(namespace, "_resume_ckpt_stage_key_impl")(
            source_node, target_node, ckpt_dir, source_location)

    def _resume_fit_deps():
        return _ns(namespace, "_build_resume_fit_deps")(namespace)

    def _resume_location_for_target(source_loc: dict, target_node: str) -> dict:
        return _ns(namespace, "_resume_location_for_target_impl")(
            source_loc,
            target_node,
            deps=_ns(namespace, "_resume_fit_deps")(),
        )

    def _resume_stage_source_for_target(
        task: dict,
        target_node: str,
        locations: Optional[list] = None,
    ) -> Optional[dict]:
        return _ns(namespace, "_resume_stage_source_for_target_impl")(
            task,
            target_node,
            locations,
            deps=_ns(namespace, "_resume_fit_deps")(),
        )

    def _resume_checkpoint_stage_check(
        task: dict,
        target_node: str,
        locations: Optional[list] = None,
    ) -> tuple:
        return _ns(namespace, "_resume_checkpoint_stage_check_impl")(
            task,
            target_node,
            locations,
            deps=_ns(namespace, "_resume_fit_deps")(),
        )

    def _task_wait_eta_seconds(task: dict) -> int:
        return _ns(namespace, "_task_wait_eta_seconds_impl")(task)

    def _resume_ckpt_stage_estimate_seconds(
        stage_state: str,
        source_loc: Optional[dict],
    ) -> int:
        return _ns(namespace, "_resume_ckpt_stage_estimate_seconds_impl")(
            stage_state,
            source_loc,
            deps=_ns(namespace, "_resume_fit_deps")(),
        )

    def _running_tasks_for_node(state: dict, node_name: str, gpu_idx=None) -> list:
        return _ns(namespace, "_running_tasks_for_node_impl")(state, node_name, gpu_idx)

    def _task_vram_pressure_mb(task: dict) -> int:
        return _ns(namespace, "_task_vram_pressure_mb_impl")(task)

    def _estimate_gpu_fit_wait_seconds(
        task: dict,
        gpu_state: dict,
        node_info: dict,
        running_gpu_tasks: list,
    ) -> Optional[int]:
        return _ns(namespace, "_estimate_gpu_fit_wait_seconds_impl")(
            task,
            gpu_state,
            node_info,
            running_gpu_tasks,
            deps=_ns(namespace, "_resume_fit_deps")(),
        )

    def _estimate_node_fit_wait_seconds(
        task: dict,
        node_state: dict,
        state: dict,
    ) -> Optional[int]:
        return _ns(namespace, "_estimate_node_fit_wait_seconds_impl")(
            task,
            node_state,
            state,
            deps=_ns(namespace, "_resume_fit_deps")(),
        )

    def _checkpoint_migration_extra_allowed_nodes(
        task: dict,
        nodes: list,
        resume_locations: list,
        state: dict,
    ) -> tuple:
        return _ns(namespace, "_checkpoint_migration_extra_allowed_nodes_impl")(
            task,
            nodes,
            resume_locations,
            state,
            deps=_ns(namespace, "_CheckpointMigrationDeps")(
                task_requires_resume_scan=_ns(namespace, "_task_requires_resume_scan"),
                estimate_node_fit_wait_seconds=_ns(namespace, "_estimate_node_fit_wait_seconds"),
                blocked_nodes_for_task=_ns(namespace, "_blocked_nodes_for_task"),
                launch_failed_nodes_for_task=_ns(namespace, "_launch_failed_nodes_for_task"),
                pick_placement=_ns(namespace, "pick_placement"),
                resume_checkpoint_stage_check=_ns(namespace, "_resume_checkpoint_stage_check"),
                resume_ckpt_stage_estimate_seconds=_ns(namespace, "_resume_ckpt_stage_estimate_seconds"),
                min_wait_s=_ns(namespace, "RESUME_CKPT_MIGRATE_MIN_WAIT_S"),
                safety_s=_ns(namespace, "RESUME_CKPT_MIGRATE_SAFETY_S"),
            ),
        )

    def _record_staged_resume_location(
        task: dict,
        target_node: str,
        source_loc: Optional[dict],
    ) -> None:
        return _ns(namespace, "_record_staged_resume_location_impl")(
            task,
            target_node,
            source_loc,
            deps=_ns(namespace, "_resume_fit_deps")(),
        )

    def _stage_for_migration(
        task: dict,
        target_node: str,
        max_ckpt_mb: int = None,
    ) -> tuple:
        if max_ckpt_mb is None:
            max_ckpt_mb = _ns(namespace, "MIGRATION_MAX_CKPT_SIZE_MB")
        return _ns(namespace, "_stage_for_migration_impl")(
            task,
            target_node,
            max_ckpt_mb=max_ckpt_mb,
            deps=_ns(namespace, "_migration_staging_deps")(),
        )

    def _migration_policy_deps():
        return _ns(namespace, "_build_migration_policy_deps")(namespace)

    def _identify_migration_candidates(
        state: dict,
        nodes: list,
        max_candidates: int = 2,
    ) -> list:
        return _ns(namespace, "_identify_migration_candidates_impl")(
            state,
            nodes,
            max_candidates=max_candidates,
            deps=_ns(namespace, "_migration_policy_deps")(),
        )

    def _stage_migration_candidates_outside_lock(
        max_candidates: int = 2,
        nodes: list | None = None,
    ):
        time_mod = _ns(namespace, "time")
        return _ns(namespace, "_stage_migration_candidates_outside_lock_impl")(
            max_candidates=max_candidates,
            nodes=nodes,
            deps=_ns(namespace, "_MigrationCandidateStagingDeps")(
                state_lock=_ns(namespace, "state_lock"),
                load_state=_ns(namespace, "load_state"),
                probe_all=_ns(namespace, "probe_all"),
                identify_migration_candidates=_ns(namespace, "_identify_migration_candidates"),
                stage_for_migration=_ns(namespace, "_stage_for_migration"),
                record_staging_failure=_ns(namespace, "_record_staging_failure"),
                notify=_ns(namespace, "notify"),
                staged_tasks=_ns(namespace, "_STAGED_TASKS"),
                staged_tasks_max=_ns(namespace, "_STAGED_TASKS_MAX"),
                now=time_mod.time,
            ),
        )

    def _consider_migration(state: dict, nodes: list, loads: Optional[dict] = None) -> list:
        return _ns(namespace, "_consider_migration_impl")(
            state,
            nodes,
            loads=loads,
            deps=_ns(namespace, "_migration_policy_deps")(),
        )

    return {
        "_migration_staging_deps": _migration_staging_deps,
        "_resume_ckpt_stage_key": _resume_ckpt_stage_key,
        "_resume_fit_deps": _resume_fit_deps,
        "_resume_location_for_target": _resume_location_for_target,
        "_resume_stage_source_for_target": _resume_stage_source_for_target,
        "_resume_checkpoint_stage_check": _resume_checkpoint_stage_check,
        "_task_wait_eta_seconds": _task_wait_eta_seconds,
        "_resume_ckpt_stage_estimate_seconds": _resume_ckpt_stage_estimate_seconds,
        "_running_tasks_for_node": _running_tasks_for_node,
        "_task_vram_pressure_mb": _task_vram_pressure_mb,
        "_estimate_gpu_fit_wait_seconds": _estimate_gpu_fit_wait_seconds,
        "_estimate_node_fit_wait_seconds": _estimate_node_fit_wait_seconds,
        "_checkpoint_migration_extra_allowed_nodes": _checkpoint_migration_extra_allowed_nodes,
        "_record_staged_resume_location": _record_staged_resume_location,
        "_stage_for_migration": _stage_for_migration,
        "_migration_policy_deps": _migration_policy_deps,
        "_identify_migration_candidates": _identify_migration_candidates,
        "_stage_migration_candidates_outside_lock": _stage_migration_candidates_outside_lock,
        "_consider_migration": _consider_migration,
    }
