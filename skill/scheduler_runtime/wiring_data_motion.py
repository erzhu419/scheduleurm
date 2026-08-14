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


def build_resume_scan_deps(namespace: Mapping[str, Any]) -> ResumeScanDeps:
    time_mod = _ns(namespace, "time")
    return ResumeScanDeps(
        node_configs=_ns(namespace, "NODES"),
        ckpt_exts=_ns(namespace, "CKPT_EXTS"),
        resume_safe_name_re=_ns(namespace, "RESUME_SAFE_NAME_RE"),
        resume_unsafe_name_re=_ns(namespace, "RESUME_UNSAFE_NAME_RE"),
        resume_scan_ttl_s=_ns(namespace, "RESUME_SCAN_TTL_S"),
        resume_scan_workers=_ns(namespace, "RESUME_SCAN_WORKERS"),
        task_gpu_capability=_ns(namespace, "_task_gpu_capability"),
        node_is_windows=_ns(namespace, "_node_is_windows"),
        windows_path_for_node=_ns(namespace, "_windows_path_for_node"),
        ps_quote=_ns(namespace, "_ps_quote"),
        run_windows_ps=_ns(namespace, "_run_windows_ps"),
        remote_path_for_node=_ns(namespace, "_remote_path_for_node"),
        run_on=_ns(namespace, "run_on"),
        cmd_has_resume_flag=_ns(namespace, "_cmd_has_resume_flag"),
        now=time_mod.time,
    )


def build_docker_wrap_deps(namespace: Mapping[str, Any]) -> DockerWrapDeps:
    return DockerWrapDeps(
        env_deploy=_ns(namespace, "env_deploy"),
        node_configs=_ns(namespace, "NODES"),
        run_on=_ns(namespace, "run_on"),
        conda_sync_ok=_ns(namespace, "_conda_sync_ok"),
    )


def build_launch_cwd_staging_deps(namespace: Mapping[str, Any]) -> LaunchCwdStagingDeps:
    return LaunchCwdStagingDeps(
        node_configs=_ns(namespace, "NODES"),
        launch_max_cwd_size_mb=_ns(namespace, "LAUNCH_MAX_CWD_SIZE_MB"),
        staging_cache_hit=_ns(namespace, "_staging_cache_hit"),
        mark_stage_success=_ns(namespace, "_mark_launch_stage_success"),
        mark_cap_exceeded=_ns(namespace, "_mark_launch_stage_cap_exceeded"),
        node_is_windows=_ns(namespace, "_node_is_windows"),
        is_windows_native_path=_ns(namespace, "_is_windows_native_path"),
        windows_path_for_node=_ns(namespace, "_windows_path_for_node"),
        run_windows_ps=_ns(namespace, "_run_windows_ps"),
        ps_quote=_ns(namespace, "_ps_quote"),
        stage_local_dir_to_windows=_ns(namespace, "_stage_local_dir_to_windows"),
        remote_path_for_node=_ns(namespace, "_remote_path_for_node"),
        run_on=_ns(namespace, "run_on"),
        stage_code_tar_for_launch=_ns(namespace, "_stage_code_tar_for_launch"),
        relay_node_for_node=_ns(namespace, "_relay_node_for_node"),
        relay_path_for_node=_ns(namespace, "_relay_path_for_node"),
        rsync_path_for_node=_ns(namespace, "_rsync_path_for_node"),
        ssh_rsync_shell_for_node=_ns(namespace, "_ssh_rsync_shell_for_node"),
        relay_ssh_target_for_node=_ns(namespace, "_relay_ssh_target_for_node"),
        run_control_subprocess=_ns(namespace, "_run_control_subprocess"),
    )


def build_resume_ckpt_staging_deps(namespace: Mapping[str, Any]) -> ResumeCkptStagingDeps:
    return ResumeCkptStagingDeps(
        node_configs=_ns(namespace, "NODES"),
        task_requires_resume_scan=_ns(namespace, "_task_requires_resume_scan"),
        resume_checkpoint_stage_check=_ns(namespace, "_resume_checkpoint_stage_check"),
        resume_ckpt_stage_key=_ns(namespace, "_resume_ckpt_stage_key"),
        staging_cache_hit=_ns(namespace, "_staging_cache_hit"),
        mark_stage_success=_ns(namespace, "_mark_launch_stage_success"),
        mark_cap_exceeded=_ns(namespace, "_mark_launch_stage_cap_exceeded"),
        node_is_windows=_ns(namespace, "_node_is_windows"),
        remote_path_for_node=_ns(namespace, "_remote_path_for_node"),
        run_on=_ns(namespace, "run_on"),
        stage_local_dir_to_windows=_ns(namespace, "_stage_local_dir_to_windows"),
        windows_path_for_node=_ns(namespace, "_windows_path_for_node"),
        run_windows_ps=_ns(namespace, "_run_windows_ps"),
        ps_quote=_ns(namespace, "_ps_quote"),
        relay_node_for_node=_ns(namespace, "_relay_node_for_node"),
        relay_path_for_node=_ns(namespace, "_relay_path_for_node"),
        rsync_path_for_node=_ns(namespace, "_rsync_path_for_node"),
        ssh_rsync_shell_for_node=_ns(namespace, "_ssh_rsync_shell_for_node"),
        relay_ssh_target_for_node=_ns(namespace, "_relay_ssh_target_for_node"),
        rsync_shell_for_pair=_ns(namespace, "_rsync_shell_for_pair"),
        run_subprocess=_ns(namespace, "_run_control_subprocess"),
    )


def build_migration_staging_deps(namespace: Mapping[str, Any]) -> MigrationStagingDeps:
    return MigrationStagingDeps(
        node_configs=_ns(namespace, "NODES"),
        migration_max_cwd_size_mb=_ns(namespace, "MIGRATION_MAX_CWD_SIZE_MB"),
        staging_cache_hit=_ns(namespace, "_staging_cache_hit"),
        mark_staging_success=_ns(namespace, "_mark_staging_cache_success"),
        remote_path_for_node=_ns(namespace, "_remote_path_for_node"),
        rsync_path_for_node=_ns(namespace, "_rsync_path_for_node"),
        rsync_shell_for_pair=_ns(namespace, "_rsync_shell_for_pair"),
        run_on=_ns(namespace, "run_on"),
        apply_node_cmd_rewrites=_ns(namespace, "_apply_node_cmd_rewrites"),
        run_subprocess=_ns(namespace, "_run_control_subprocess"),
    )


def build_launch_staging_plan_deps(namespace: Mapping[str, Any]) -> LaunchStagingPlanDeps:
    time_mod = _ns(namespace, "time")
    cap_exceeded = _ns(namespace, "_STAGING_CAP_EXCEEDED")
    staging_ttl_s = _ns(namespace, "STAGING_TTL_S")

    def _staging_cap_recent(key: tuple) -> bool:
        cap_ts = cap_exceeded.get(key)
        return bool(cap_ts is not None and (time_mod.time() - cap_ts) <= staging_ttl_s)

    return LaunchStagingPlanDeps(
        node_configs=_ns(namespace, "NODES"),
        queued_wait_for_file_block_reason=_ns(namespace, "_queued_wait_for_file_block_reason"),
        hpc_cpu_pool_soft_require_nodes=_ns(namespace, "_hpc_cpu_pool_soft_require_nodes"),
        task_requires_resume_scan=_ns(namespace, "_task_requires_resume_scan"),
        node_is_windows=_ns(namespace, "_node_is_windows"),
        node_cpu_fallback_block_reason=_ns(namespace, "_node_cpu_fallback_block_reason"),
        staging_cache_hit=_ns(namespace, "_staging_cache_hit"),
        staging_cap_recent=_staging_cap_recent,
        resume_checkpoint_stage_check=_ns(namespace, "_resume_checkpoint_stage_check"),
    )


def build_resume_fit_deps(namespace: Mapping[str, Any]) -> ResumeFitDeps:
    time_mod = _ns(namespace, "time")
    return ResumeFitDeps(
        node_configs=_ns(namespace, "NODES"),
        staging_cap_exceeded=_ns(namespace, "_STAGING_CAP_EXCEEDED"),
        staging_fails=_ns(namespace, "_STAGING_FAILS"),
        staging_ttl_s=_ns(namespace, "STAGING_TTL_S"),
        staging_fail_cooldown_s=_ns(namespace, "STAGING_FAIL_COOLDOWN_S"),
        resume_ckpt_migrate_overhead_s=_ns(namespace, "RESUME_CKPT_MIGRATE_OVERHEAD_S"),
        resume_ckpt_migrate_est_mbps=_ns(namespace, "RESUME_CKPT_MIGRATE_EST_MBPS"),
        remote_path_for_node=_ns(namespace, "_remote_path_for_node"),
        staging_cache_hit=_ns(namespace, "_staging_cache_hit"),
        gpu_fits=_ns(namespace, "_gpu_fits"),
        pick_placement=_ns(namespace, "pick_placement"),
        task_required_gpu_idx=_ns(namespace, "_task_required_gpu_idx"),
        compute_node_load_seconds=_ns(namespace, "compute_node_load_seconds"),
        now=time_mod.time,
    )


def build_migration_policy_deps(namespace: Mapping[str, Any]) -> MigrationPolicyDeps:
    time_mod = _ns(namespace, "time")
    return MigrationPolicyDeps(
        compute_node_load_seconds=_ns(namespace, "compute_node_load_seconds"),
        blocked_nodes_for_task=_ns(namespace, "_blocked_nodes_for_task"),
        launch_failed_nodes_for_task=_ns(namespace, "_launch_failed_nodes_for_task"),
        staging_recently_failed=_ns(namespace, "_staging_recently_failed"),
        can_migrate_to=_ns(namespace, "_can_migrate_to"),
        migration_free_threshold_s=_ns(namespace, "MIGRATION_FREE_THRESHOLD_S"),
        migration_load_ratio=_ns(namespace, "MIGRATION_LOAD_RATIO"),
        migration_min_source_load_s=_ns(namespace, "MIGRATION_MIN_SOURCE_LOAD_S"),
        migration_min_task_eta_s=_ns(namespace, "MIGRATION_MIN_TASK_ETA_S"),
        migration_cooldown_s=_ns(namespace, "MIGRATION_COOLDOWN_S"),
        migration_max_per_dispatch=_ns(namespace, "MIGRATION_MAX_PER_DISPATCH"),
        now=time_mod.time,
    )


def build_result_sync_executor_deps(namespace: Mapping[str, Any]) -> ResultSyncExecutorDeps:
    return ResultSyncExecutorDeps(
        node_is_windows=_ns(namespace, "_node_is_windows"),
        sync_windows_result=_ns(namespace, "_sync_windows_result"),
        remote_path_for_node=_ns(namespace, "_remote_path_for_node"),
        relay_node_for_node=_ns(namespace, "_relay_node_for_node"),
        relay_path_for_node=_ns(namespace, "_relay_path_for_node"),
        relay_ssh_target_for_node=_ns(namespace, "_relay_ssh_target_for_node"),
        run_on=_ns(namespace, "run_on"),
        ssh_rsync_shell_for_node=_ns(namespace, "_ssh_rsync_shell_for_node"),
        rsync_path_for_node=_ns(namespace, "_rsync_path_for_node"),
        ssh_target_for_node=_ns(namespace, "_ssh_target_for_node"),
        result_sync_timeout_s=_ns(namespace, "RESULT_SYNC_TIMEOUT_S"),
    )
