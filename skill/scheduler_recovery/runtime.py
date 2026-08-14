"""Runtime-bound node-reporting/recovery/preload wrappers for scheduler.py."""

from __future__ import annotations

from typing import Any, Mapping, Optional


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_recovery_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    def _node_display_name(name: str) -> str:
        return _ns(namespace, "_canonical_node_name")(name)

    def _node_reporting_deps():
        time_mod = _ns(namespace, "time")
        return _ns(namespace, "_NodeReportingDeps")(
            node_configs=_ns(namespace, "NODES"),
            node_display_name=_ns(namespace, "_node_display_name"),
            format_mem_gb=_ns(namespace, "_format_mem_gb"),
            format_node_ram_summary=_ns(namespace, "_format_node_ram_summary"),
            scheduler_id=_ns(namespace, "_ClaimManager").scheduler_id,
            cpu_ownership_snapshot=_ns(namespace, "_cpu_ownership_snapshot"),
            now=time_mod.time,
            print_fn=print,
        )

    def _node_summary_visible(node: dict) -> bool:
        return _ns(namespace, "_node_summary_visible_impl")(node)

    def _node_summary_sort_key(node: dict) -> tuple:
        return _ns(namespace, "_node_summary_sort_key_impl")(
            node,
            deps=_ns(namespace, "_node_reporting_deps")(),
        )

    def _iter_node_summary_nodes(nodes):
        return _ns(namespace, "_iter_node_summary_nodes_impl")(
            nodes,
            deps=_ns(namespace, "_node_reporting_deps")(),
        )

    def _print_node_summary(nodes):
        return _ns(namespace, "_print_node_summary_impl")(
            nodes,
            deps=_ns(namespace, "_node_reporting_deps")(),
        )

    def _format_task_location(task):
        legacy_bucket = None
        if task.get("slurm_job_id"):
            legacy_bucket = "gpu" if int(task.get("est_vram_mb") or 0) > 0 else "cpu"
        return _ns(namespace, "_format_task_location_impl")(task, legacy_bucket=legacy_bucket)

    def _claim_wait_s(claim: dict) -> int:
        return _ns(namespace, "_claim_wait_s_impl")(
            claim,
            deps=_ns(namespace, "_node_reporting_deps")(),
        )

    def _format_claim_record(claim: dict) -> str:
        return _ns(namespace, "_format_claim_record_impl")(
            claim,
            deps=_ns(namespace, "_node_reporting_deps")(),
        )

    def _format_node_claim_summary(node: dict) -> str:
        return _ns(namespace, "_format_node_claim_summary_impl")(
            node,
            deps=_ns(namespace, "_node_reporting_deps")(),
        )

    def _format_claim_intent_hint_for_task(task: dict, node: str, snap: dict) -> str:
        return _ns(namespace, "_format_claim_intent_hint_for_task_impl")(
            task,
            node,
            snap,
            deps=_ns(namespace, "_node_reporting_deps")(),
        )

    def _local_orphan_recovery_deps():
        return _ns(namespace, "_build_local_orphan_recovery_deps")(namespace)

    def _adopt_live_local_pid_rows(
        task: dict,
        node: str,
        rows: list,
        prior_status: Optional[str] = None,
    ) -> bool:
        return _ns(namespace, "_adopt_live_local_pid_rows_impl")(
            task,
            node,
            rows,
            deps=_ns(namespace, "_local_orphan_recovery_deps")(),
            prior_status=prior_status,
        )

    def _try_recover_orphan_local_task(task: dict, node: str) -> bool:
        return _ns(namespace, "_try_recover_orphan_local_task_impl")(
            task,
            node,
            deps=_ns(namespace, "_local_orphan_recovery_deps")(),
        )

    def _scan_scheduleurm_env_pids_on_node(node: str) -> dict:
        return _ns(namespace, "_scan_scheduleurm_env_pids_on_node_impl")(
            node,
            deps=_ns(namespace, "_local_orphan_recovery_deps")(),
        )

    def _scan_launch_recovery_evidence_on_node(node: str, tasks: list[dict]) -> dict:
        return _ns(namespace, "_scan_launch_recovery_evidence_on_node_impl")(
            node,
            tasks,
            deps=_ns(namespace, "_local_orphan_recovery_deps")(),
        )

    def recover_queued_live_local_tasks(state: dict) -> int:
        return _ns(namespace, "_recover_queued_live_local_tasks_impl")(
            state,
            deps=_ns(namespace, "_local_orphan_recovery_deps")(),
        )

    def _terminal_finalize_deps():
        return _ns(namespace, "_build_terminal_finalize_deps")(namespace)

    def _try_finalize_terminal_local_task(task: dict, node: str, state: dict) -> bool:
        return _ns(namespace, "_try_finalize_terminal_local_task_impl")(
            task,
            node,
            state,
            deps=_ns(namespace, "_terminal_finalize_deps")(),
        )

    def _launch_recovery_deps():
        return _ns(namespace, "_build_launch_recovery_deps")(namespace)

    def recover_stale_launching_tasks(
        state,
        now: Optional[float] = None,
        reset_s: int = None,
    ) -> int:
        if reset_s is None:
            reset_s = _ns(namespace, "LAUNCHING_RESET_S")
        return _ns(namespace, "_recover_stale_launching_tasks_impl")(
            state,
            reset_s=reset_s,
            now=now,
            deps=_ns(namespace, "_launch_recovery_deps")(),
        )

    def _stale_launch_recovery_deps():
        time_mod = _ns(namespace, "time")
        claim_manager = _ns(namespace, "_ClaimManager")
        return _ns(namespace, "_StaleLaunchRecoveryDeps")(
            state_lock=_ns(namespace, "state_lock"),
            load_state=_ns(namespace, "load_state"),
            save_state=_ns(namespace, "save_state"),
            scan_node_evidence=_ns(namespace, "_scan_launch_recovery_evidence_on_node"),
            adopt_live_pid_rows=_ns(namespace, "_adopt_live_local_pid_rows"),
            node_is_windows=_ns(namespace, "_node_is_windows"),
            claim_enabled_for=claim_manager.enabled_for,
            release_task_claims_and_intents=_ns(namespace, "_release_task_claims_and_intents"),
            clear_live_eta_fields=_ns(namespace, "_clear_live_eta_fields"),
            set_current_usage=_ns(namespace, "_set_current_usage"),
            notify=_ns(namespace, "notify"),
            now=time_mod.time,
            max_workers=_ns(namespace, "PROBE_ROUTE_MAX_WORKERS"),
        )

    def recover_stale_launching_tasks_outside_lock(
        *,
        purpose: str = "launch-recovery",
        lock_timeout=None,
        reset_s: int = None,
    ) -> int:
        if reset_s is None:
            reset_s = _ns(namespace, "LAUNCHING_RESET_S")
        return _ns(namespace, "_recover_stale_launching_tasks_outside_lock_impl")(
            reset_s=reset_s,
            deps=_ns(namespace, "_stale_launch_recovery_deps")(),
            purpose=purpose,
            lock_timeout=lock_timeout,
        )

    def _wait_for_deps():
        return _ns(namespace, "_build_wait_for_deps")(namespace)

    def cmd_wait_for(args):
        return _ns(namespace, "_cmd_wait_for_impl")(
            args,
            deps=_ns(namespace, "_wait_for_deps")(),
        )

    def _preload_docker_images_outside_lock(task_ids: Optional[set] = None):
        return _ns(namespace, "_preload_docker_images_outside_lock_impl")(
            task_ids=task_ids,
            deps=_ns(namespace, "_DockerPreloadDeps")(
                env_deploy=_ns(namespace, "env_deploy"),
                load_state=_ns(namespace, "load_state"),
                node_configs=_ns(namespace, "NODES"),
                node_is_windows=_ns(namespace, "_node_is_windows"),
                run_on=_ns(namespace, "run_on"),
                notify=_ns(namespace, "notify"),
                record_conda_sync_ok=_ns(namespace, "_record_conda_sync_ok"),
                record_conda_sync_failed=_ns(namespace, "_record_conda_sync_failed"),
            ),
        )

    return {
        "_node_display_name": _node_display_name,
        "_node_reporting_deps": _node_reporting_deps,
        "_node_summary_visible": _node_summary_visible,
        "_node_summary_sort_key": _node_summary_sort_key,
        "_iter_node_summary_nodes": _iter_node_summary_nodes,
        "_print_node_summary": _print_node_summary,
        "_format_task_location": _format_task_location,
        "_claim_wait_s": _claim_wait_s,
        "_format_claim_record": _format_claim_record,
        "_format_node_claim_summary": _format_node_claim_summary,
        "_format_claim_intent_hint_for_task": _format_claim_intent_hint_for_task,
        "_local_orphan_recovery_deps": _local_orphan_recovery_deps,
        "_adopt_live_local_pid_rows": _adopt_live_local_pid_rows,
        "_try_recover_orphan_local_task": _try_recover_orphan_local_task,
        "_scan_scheduleurm_env_pids_on_node": _scan_scheduleurm_env_pids_on_node,
        "_scan_launch_recovery_evidence_on_node": _scan_launch_recovery_evidence_on_node,
        "recover_queued_live_local_tasks": recover_queued_live_local_tasks,
        "_terminal_finalize_deps": _terminal_finalize_deps,
        "_try_finalize_terminal_local_task": _try_finalize_terminal_local_task,
        "_launch_recovery_deps": _launch_recovery_deps,
        "recover_stale_launching_tasks": recover_stale_launching_tasks,
        "_stale_launch_recovery_deps": _stale_launch_recovery_deps,
        "recover_stale_launching_tasks_outside_lock": recover_stale_launching_tasks_outside_lock,
        "_wait_for_deps": _wait_for_deps,
        "cmd_wait_for": cmd_wait_for,
        "_preload_docker_images_outside_lock": _preload_docker_images_outside_lock,
    }
