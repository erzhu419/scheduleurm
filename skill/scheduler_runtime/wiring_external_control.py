"""Runtime dependency wiring helpers grouped by subsystem."""

from __future__ import annotations
import builtins
from contextlib import contextmanager
from typing import Any, Mapping
from scheduler_adopt.probe import AdoptProbeDeps
from scheduler_claim.tending import ClaimTendingDeps
from scheduler_environment.docker_wrap import DockerWrapDeps
from scheduler_environment.env_smoke import EnvSmokeDeps
from scheduler_placement_engine.eviction import EvictionDeps
from scheduler_external.reconcile import ExternalReconcileDeps
from scheduler_launch.commit import LaunchCommitDeps
from scheduler_staging.launch_cwd import LaunchCwdStagingDeps
from scheduler_launch.recovery import LaunchRecoveryDeps
from scheduler_launch.staging_plan import LaunchStagingPlanDeps
from scheduler_backend.local_orphan_recovery import LocalOrphanRecoveryDeps
from scheduler_migration.policy import MigrationPolicyDeps
from scheduler_migration.staging import MigrationStagingDeps
from scheduler_node.gates import NodeResourceGateDeps
from scheduler_notification.notify import NotifyDeps
from scheduler_placement_engine.core import PlacementRuntime
from scheduler_placement_engine.post_dispatch_thresholds import PostDispatchThresholdDeps
from scheduler_placement_engine.preemption import PreemptionDeps
from scheduler_resource.accounting import ResourceAccountingDeps
from scheduler_result_sync.executor import ResultSyncExecutorDeps
from scheduler_staging.resume_ckpt import ResumeCkptStagingDeps
from scheduler_migration.resume_fit import ResumeFitDeps
from scheduler_migration.resume_scan import ResumeScanDeps
from scheduler_failure.terminal_finalize import TerminalFinalizeDeps
from scheduler_commands.wait_for import WaitForDeps


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_wait_for_deps(namespace: Mapping[str, Any]) -> WaitForDeps:
    time_mod = _ns(namespace, "time")

    @contextmanager
    def refresh_writer_lease():
        acquired = False
        try:
            with _ns(namespace, "watcher_lifetime_lock")(
                timeout_s=0,
                purpose="wait-for:refresh-owner",
            ):
                acquired = True
                yield True
        except _ns(namespace, "SchedulerLockTimeout"):
            if acquired:
                raise
            yield False

    return WaitForDeps(
        state_lock=_ns(namespace, "state_lock"),
        load_state=_ns(namespace, "load_state"),
        save_state=_ns(namespace, "save_state"),
        recover_stale_launching_tasks=_ns(namespace, "recover_stale_launching_tasks"),
        update_running_tasks=_ns(namespace, "update_running_tasks"),
        running_probe_snapshot_outside_lock=_ns(namespace, "_running_probe_snapshot_outside_lock"),
        eta_tail_snapshot_outside_lock=_ns(namespace, "_eta_tail_snapshot_outside_lock"),
        refresh_writer_lease=refresh_writer_lease,
        now=time_mod.time,
        sleep=time_mod.sleep,
        print_fn=builtins.print,
    )


def build_notify_deps(namespace: Mapping[str, Any]) -> NotifyDeps:
    time_mod = _ns(namespace, "time")
    return NotifyDeps(
        watcher_log=_ns(namespace, "WATCHER_LOG"),
        watcher_log_max_mb=_ns(namespace, "WATCHER_LOG_MAX_MB"),
        watcher_log_generations=_ns(namespace, "WATCHER_LOG_GENERATIONS"),
        feishu_config=_ns(namespace, "FEISHU_CONFIG"),
        maybe_rotate_log=_ns(namespace, "maybe_rotate_log"),
        format_task_location=_ns(namespace, "_format_task_location"),
        format_mem_gb=_ns(namespace, "_format_mem_gb"),
        max_launch_retry=_ns(namespace, "MAX_LAUNCH_RETRY"),
        now=time_mod.time,
    )


def build_adopt_probe_deps(namespace: Mapping[str, Any]) -> AdoptProbeDeps:
    return AdoptProbeDeps(
        node_configs=_ns(namespace, "NODES"),
        node_is_windows=_ns(namespace, "_node_is_windows"),
        run_on=_ns(namespace, "run_on"),
        local_user=_ns(namespace, "_local_user"),
        task_pids=_ns(namespace, "_task_pids"),
        descendants_cap=_ns(namespace, "_DESCENDANTS_CAP"),
    )


def build_external_reconcile_deps(namespace: Mapping[str, Any]) -> ExternalReconcileDeps:
    time_mod = _ns(namespace, "time")
    return ExternalReconcileDeps(
        collect_external_task_probe_data=_ns(namespace, "_collect_external_task_probe_data"),
        local_user=_ns(namespace, "_local_user"),
        task_pids=_ns(namespace, "_task_pids"),
        descendants_of=_ns(namespace, "_descendants_of"),
        is_scheduler_control_process=_ns(namespace, "_is_scheduler_control_process"),
        refresh_adopted_resources=_ns(namespace, "_refresh_adopted_resources"),
        infer_adopt_cwd_from_cmdline=_ns(namespace, "_infer_adopt_cwd_from_cmdline"),
        path_under_roots=_ns(namespace, "_path_under_roots"),
        project_from_path=_ns(namespace, "_project_from_path"),
        history_get=_ns(namespace, "history_get"),
        allocate_task_id=_ns(namespace, "_allocate_task_id"),
        default_ram_mb=_ns(namespace, "DEFAULT_RAM_MB"),
        default_vram_mb=_ns(namespace, "DEFAULT_VRAM_MB"),
        now=time_mod.time,
    )


def build_resource_accounting_deps(namespace: Mapping[str, Any]) -> ResourceAccountingDeps:
    claim_manager = _ns(namespace, "_ClaimManager")
    return ResourceAccountingDeps(
        scheduler_id=claim_manager.scheduler_id,
        reserved_cpu_slots_for_task=_ns(namespace, "_reserved_cpu_slots_for_task"),
        node_physical_cores=_ns(namespace, "_node_physical_cores"),
        algorithm_name=_ns(namespace, "_algorithm_name"),
        algorithm_config_snapshot=_ns(namespace, "_algorithm_config_snapshot"),
    )


def build_env_smoke_deps(namespace: Mapping[str, Any]) -> EnvSmokeDeps:
    return EnvSmokeDeps(
        load_state=_ns(namespace, "load_state"),
        notify=_ns(namespace, "notify"),
        node_is_windows=_ns(namespace, "_node_is_windows"),
        apply_node_cmd_rewrites=_ns(namespace, "_apply_node_cmd_rewrites"),
        run_on=_ns(namespace, "run_on"),
        environ=_ns(namespace, "os").environ,
    )


def build_claim_tending_deps(namespace: Mapping[str, Any]) -> ClaimTendingDeps:
    claim_manager = _ns(namespace, "_ClaimManager")
    return ClaimTendingDeps(
        enabled_for_node=claim_manager.enabled_for,
        collect_claim_renewal_inputs=_ns(namespace, "_collect_claim_renewal_inputs"),
        claim_record_for_task=_ns(namespace, "_claim_resource_record_for_task"),
        scheduler_id=claim_manager.scheduler_id,
        enumerate_claims=claim_manager.enumerate,
        stale_claim_task_ids=_ns(namespace, "_stale_claim_task_ids"),
        release_claim=claim_manager.release,
        renew_many=claim_manager.renew_many,
        claim_pid_updates=_ns(namespace, "_claim_pid_updates"),
        update_pid=claim_manager.update_pid,
        gc_stale=claim_manager.gc_stale,
        notify=_ns(namespace, "notify"),
    )
