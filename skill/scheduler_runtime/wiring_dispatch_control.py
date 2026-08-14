"""Runtime dependency wiring helpers grouped by subsystem."""

from __future__ import annotations
import builtins
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


def build_post_dispatch_threshold_deps(namespace: Mapping[str, Any]) -> PostDispatchThresholdDeps:
    time_mod = _ns(namespace, "time")
    return PostDispatchThresholdDeps(
        node_configs=_ns(namespace, "NODES"),
        default_cpu_cores=_ns(namespace, "DEFAULT_CPU_CORES"),
        gpu_evict_stable_progress_min_age_s=_ns(namespace, "GPU_EVICT_STABLE_PROGRESS_MIN_AGE_S"),
        gpu_evict_rollback_max_age_s=_ns(namespace, "GPU_EVICT_ROLLBACK_MAX_AGE_S"),
        local_gpu_host_cpu_evict_pct=_ns(namespace, "LOCAL_GPU_HOST_CPU_EVICT_PCT"),
        effective_elapsed_s=_ns(namespace, "_effective_elapsed_s"),
        task_has_progress_evidence=_ns(namespace, "_task_has_progress_evidence"),
        task_progress_ratio=_ns(namespace, "_task_progress_ratio"),
        task_ram_pressure_mb=_ns(namespace, "_task_ram_pressure_mb"),
        is_slurm_managed=_ns(namespace, "_is_legacy_external_managed"),
        task_evict_loss_protected=_ns(namespace, "_task_evict_loss_protected"),
        ckpt_task_evict_protected=_ns(namespace, "_ckpt_task_evict_protected"),
        task_ignores_one_third_pack_rule=_ns(namespace, "_task_ignores_one_third_pack_rule"),
        local_cpu_pressure_high=_ns(namespace, "_local_cpu_pressure_high"),
        gpu_freeze_line_mb=_ns(namespace, "_gpu_freeze_line_mb"),
        gpu_threshold_snapshot=_ns(namespace, "_gpu_threshold_snapshot"),
        summarize_task_for_resource_log=_ns(namespace, "_summarize_task_for_resource_log"),
        node_ram_snapshot=_ns(namespace, "_node_ram_snapshot"),
        format_mem_gb=_ns(namespace, "_format_mem_gb"),
        evict_to_queue=_ns(namespace, "_evict_to_queue"),
        notify=_ns(namespace, "notify"),
        now=time_mod.time,
    )


def build_preemption_deps(namespace: Mapping[str, Any]) -> PreemptionDeps:
    time_mod = _ns(namespace, "time")
    return PreemptionDeps(
        node_configs=_ns(namespace, "NODES"),
        queue_wait_min=_ns(namespace, "PREEMPT_QUEUE_WAIT_MIN"),
        victim_min_age_min=_ns(namespace, "PREEMPT_VICTIM_MIN_AGE_MIN"),
        victim_max_age_min=_ns(namespace, "PREEMPT_VICTIM_MAX_AGE_MIN"),
        max_victims_per_dispatch=_ns(namespace, "PREEMPT_MAX_VICTIMS_PER_DISPATCH"),
        default_cpu_cores=_ns(namespace, "DEFAULT_CPU_CORES"),
        default_ram_mb=_ns(namespace, "DEFAULT_RAM_MB"),
        node_ram_headroom_mb=_ns(namespace, "_node_ram_headroom_mb"),
        ignore_cpu_for_server_gpu_task=_ns(namespace, "_ignore_cpu_for_server_gpu_task"),
        is_slurm_managed=_ns(namespace, "_is_legacy_external_managed"),
        task_evict_loss_protected=_ns(namespace, "_task_evict_loss_protected"),
        task_ram_pressure_mb=_ns(namespace, "_task_ram_pressure_mb"),
        task_progress_ratio=_ns(namespace, "_task_progress_ratio"),
        evict_to_queue=_ns(namespace, "_evict_to_queue"),
        format_mem_gb=_ns(namespace, "_format_mem_gb"),
        now=time_mod.time,
    )


def build_eviction_deps(namespace: Mapping[str, Any]) -> EvictionDeps:
    time_mod = _ns(namespace, "time")
    return EvictionDeps(
        release_task_claims_and_intents=_ns(namespace, "_release_task_claims_and_intents"),
        task_pids=_ns(namespace, "_task_pids"),
        record_task_kill_actor=_ns(namespace, "_record_task_kill_actor"),
        kill_task_processes=_ns(namespace, "_kill_task_processes"),
        notify=_ns(namespace, "notify"),
        set_current_usage=_ns(namespace, "_set_current_usage"),
        clear_live_eta_fields=_ns(namespace, "_clear_live_eta_fields"),
        local_cpu_pressure_high=_ns(namespace, "_local_cpu_pressure_high"),
        evict_relaunch_cooldown_s=_ns(namespace, "EVICT_RELAUNCH_COOLDOWN_S"),
        local_cpu_evict_node_cooldown_s=_ns(namespace, "LOCAL_CPU_EVICT_NODE_COOLDOWN_S"),
        local_gpu_host_cpu_block_pct=_ns(namespace, "LOCAL_GPU_HOST_CPU_BLOCK_PCT"),
        now=time_mod.time,
    )


def build_launch_commit_deps(namespace: Mapping[str, Any]) -> LaunchCommitDeps:
    return LaunchCommitDeps(
        state_lock=_ns(namespace, "state_lock"),
        load_state=_ns(namespace, "load_state"),
        save_state=_ns(namespace, "save_state"),
        apply_launch_result_to_task=_ns(namespace, "_apply_launch_result_to_task"),
        kill_task=_ns(namespace, "_BACKEND").kill,
        release_task_claims_and_intents=_ns(namespace, "_release_task_claims_and_intents"),
    )


def build_local_orphan_recovery_deps(namespace: Mapping[str, Any]) -> LocalOrphanRecoveryDeps:
    time_mod = _ns(namespace, "time")
    claim_manager = _ns(namespace, "_ClaimManager")
    return LocalOrphanRecoveryDeps(
        state_dir=_ns(namespace, "STATE_DIR"),
        node_configs=_ns(namespace, "NODES"),
        node_is_windows=_ns(namespace, "_node_is_windows"),
        run_on=_ns(namespace, "run_on"),
        remember_last_placement=_ns(namespace, "_remember_last_placement"),
        set_current_usage=_ns(namespace, "_set_current_usage"),
        claim_enabled_for=claim_manager.enabled_for,
        update_claim_pid=claim_manager.update_pid,
        requires_local_capacity_check=_ns(namespace, "_requires_local_capacity_check"),
        now=time_mod.time,
        recover_remote_nodes=_ns(namespace, "RECOVER_QUEUED_LIVE_REMOTE_NODES"),
    )


def build_terminal_finalize_deps(namespace: Mapping[str, Any]) -> TerminalFinalizeDeps:
    time_mod = _ns(namespace, "time")
    claim_manager = _ns(namespace, "_ClaimManager")
    nodes = _ns(namespace, "NODES")
    return TerminalFinalizeDeps(
        state_dir=_ns(namespace, "STATE_DIR"),
        node_is_windows=_ns(namespace, "_node_is_windows"),
        is_local_node=lambda node: nodes.get(node, {}).get("host") is None,
        run_on=_ns(namespace, "run_on"),
        diagnose_terminal=_ns(namespace, "_diagnose_terminal"),
        requeue_after_crash=_ns(namespace, "_requeue_after_crash"),
        record_result_artifacts=_ns(namespace, "_record_result_artifacts"),
        claim_enabled_for=claim_manager.enabled_for,
        release_task_claims_and_intents=_ns(namespace, "_release_task_claims_and_intents"),
        set_current_usage=_ns(namespace, "_set_current_usage"),
        now=time_mod.time,
    )


def build_launch_recovery_deps(namespace: Mapping[str, Any]) -> LaunchRecoveryDeps:
    time_mod = _ns(namespace, "time")
    claim_manager = _ns(namespace, "_ClaimManager")
    return LaunchRecoveryDeps(
        try_recover_orphan_local_task=_ns(namespace, "_try_recover_orphan_local_task"),
        try_finalize_terminal_local_task=_ns(namespace, "_try_finalize_terminal_local_task"),
        claim_enabled_for=claim_manager.enabled_for,
        release_task_claims_and_intents=_ns(namespace, "_release_task_claims_and_intents"),
        clear_live_eta_fields=_ns(namespace, "_clear_live_eta_fields"),
        now=time_mod.time,
    )
