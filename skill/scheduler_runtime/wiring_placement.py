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


def build_node_resource_gate_deps(namespace: Mapping[str, Any]) -> NodeResourceGateDeps:
    return NodeResourceGateDeps(
        default_cpu_cores=_ns(namespace, "DEFAULT_CPU_CORES"),
        default_ram_mb=_ns(namespace, "DEFAULT_RAM_MB"),
        hard_rule_bypassed=_ns(namespace, "_hard_rule_bypassed"),
        ignore_cpu_for_server_gpu_task=_ns(namespace, "_ignore_cpu_for_server_gpu_task"),
        task_is_gpu_capacity_task=_ns(namespace, "_task_is_gpu_capacity_task"),
        node_ram_headroom_mb=_ns(namespace, "_node_ram_headroom_mb"),
        local_gpu_host_cpu_block_pct=_ns(namespace, "LOCAL_GPU_HOST_CPU_BLOCK_PCT"),
    )


def build_placement_runtime(namespace: Mapping[str, Any]) -> PlacementRuntime:
    return PlacementRuntime(
        nodes_by_name=_ns(namespace, "NODES"),
        default_vram_mb=_ns(namespace, "DEFAULT_VRAM_MB"),
        vram_margin_mb=_ns(namespace, "VRAM_MARGIN_MB"),
        gpu_empty_used_mb=_ns(namespace, "GPU_EMPTY_USED_MB"),
        one_third_pack_rule=_ns(namespace, "ONE_THIRD_PACK_RULE"),
        hpc_cpu_pool_soft_require_nodes=_ns(namespace, "_hpc_cpu_pool_soft_require_nodes"),
        task_required_gpu_idx=_ns(namespace, "_task_required_gpu_idx"),
        blocked_nodes_for_task=_ns(namespace, "_blocked_nodes_for_task"),
        launch_failed_nodes_for_task=_ns(namespace, "_launch_failed_nodes_for_task"),
        algorithm_trace_enabled=_ns(namespace, "_algorithm_trace_enabled"),
        algorithm_trace_slot_id=_ns(namespace, "_algorithm_trace_slot_id"),
        algorithm_trace_candidate_row=_ns(namespace, "_algorithm_trace_candidate_row"),
        algorithm_trace_decision_slot=_ns(namespace, "_algorithm_trace_decision_slot"),
        algorithm_log_decision_slot=_ns(namespace, "_algorithm_log_decision_slot"),
        algorithm_name=_ns(namespace, "_algorithm_name"),
        algorithm_selected_gpu_audit=_ns(namespace, "_algorithm_selected_gpu_audit"),
        evict_node_cooldown_block_reason=_ns(namespace, "_evict_node_cooldown_block_reason"),
        node_is_windows=_ns(namespace, "_node_is_windows"),
        node_cpu_fallback_block_reason=_ns(namespace, "_node_cpu_fallback_block_reason"),
        node_gpu_task_block_reason=_ns(namespace, "_node_gpu_task_block_reason"),
        node_resources_ok=_ns(namespace, "_node_resources_ok"),
        node_schedulable_cpu_free=_ns(namespace, "_node_schedulable_cpu_free"),
        candidate_runtime_seconds=_ns(namespace, "_candidate_runtime_seconds"),
        gpu_task_cap_for_task=_ns(namespace, "_gpu_task_cap_for_task"),
        hard_rule_bypassed=_ns(namespace, "_hard_rule_bypassed"),
        algorithm_gpu_score=_ns(namespace, "_algorithm_gpu_score"),
        task_ignores_one_third_pack_rule=_ns(namespace, "_task_ignores_one_third_pack_rule"),
        gpu_freeze_line_mb=_ns(namespace, "_gpu_freeze_line_mb"),
        node_gpu_util_limit=_ns(namespace, "_node_gpu_util_limit"),
        algorithm_gpu_fit_block_reason=_ns(namespace, "_algorithm_gpu_fit_block_reason"),
    )
