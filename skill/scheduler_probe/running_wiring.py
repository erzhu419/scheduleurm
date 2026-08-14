from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from scheduler_claim.probe_fold import ClaimProbeFoldDeps
from scheduler_eta.refresh import EtaRefreshDeps
from scheduler_eta.background import EtaBackgroundRefreshDeps
from scheduler_eta.tail_snapshot import EtaTailOutputDeps, EtaTailSnapshotDeps
from .all import ProbeAllDeps
from .node import ProbeNodeDeps
from scheduler_remote.exec import RemoteExecDeps
from scheduler_running.lifecycle import RunningLifecycleDeps
from scheduler_running.snapshots import (
    BatchCheckRunningDeps,
    RunningProbeSnapshotDeps,
    UpdateRunningTasksDeps,
)
from .sudo_cpu import SudoCpuBatchProbeDeps
from scheduler_failure.terminal_snapshot import TerminalDiagnosticsSnapshotDeps
from scheduler_windows.host_extras import WindowsHostExtrasDeps
from scheduler_windows.probe import WindowsNodeProbeDeps


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_remote_exec_deps(namespace: Mapping[str, Any]) -> RemoteExecDeps:
    os_mod = _ns(namespace, "os")
    signal_mod = _ns(namespace, "signal")
    subprocess_mod = _ns(namespace, "subprocess")
    time_mod = _ns(namespace, "time")
    return RemoteExecDeps(
        node_configs=_ns(namespace, "NODES"),
        canonical_node_name=_ns(namespace, "_canonical_node_name"),
        node_is_windows=_ns(namespace, "_node_is_windows"),
        raise_if_watcher_shutdown=_ns(namespace, "_raise_if_watcher_shutdown"),
        register_control_subprocess=_ns(namespace, "_register_control_subprocess"),
        unregister_control_subprocess=_ns(namespace, "_unregister_control_subprocess"),
        terminate_control_process_group=_ns(namespace, "_terminate_control_process_group"),
        ssh_route_cache_ttl_s=_ns(namespace, "SSH_ROUTE_CACHE_TTL_S"),
        ssh_login_preflight_timeout_s=_ns(namespace, "SSH_LOGIN_PREFLIGHT_TIMEOUT_S"),
        ssh_login_failure_cache_ttl_s=_ns(namespace, "SSH_LOGIN_FAILURE_CACHE_TTL_S"),
        ssh_disable_mux=_ns(namespace, "SSH_DISABLE_MUX"),
        ssh_proxy_jump_cache=_ns(namespace, "_SSH_PROXY_JUMP_CACHE"),
        ssh_route_jump_cache=_ns(namespace, "_SSH_ROUTE_JUMP_CACHE"),
        ssh_outer_route_cache=_ns(namespace, "_SSH_OUTER_ROUTE_CACHE"),
        probe_ssh_proxy_jump_callback=_ns(namespace, "_probe_ssh_proxy_jump"),
        probe_outer_ssh_route_callback=_ns(namespace, "_probe_outer_ssh_route"),
        subprocess_popen=subprocess_mod.Popen,
        subprocess_run=subprocess_mod.run,
        subprocess_check_output=subprocess_mod.check_output,
        kill=os_mod.kill,
        now=time_mod.time,
        expanduser=os_mod.path.expanduser,
        pipe=subprocess_mod.PIPE,
        devnull=subprocess_mod.DEVNULL,
        sigkill=signal_mod.SIGKILL,
        sigterm=signal_mod.SIGTERM,
        route_probe_max_workers=_ns(namespace, "PROBE_ROUTE_MAX_WORKERS"),
    )


def build_windows_host_extras_deps(namespace: Mapping[str, Any]) -> WindowsHostExtrasDeps:
    subprocess_mod = _ns(namespace, "subprocess")
    return WindowsHostExtrasDeps(
        path_exists=lambda path: Path(path).exists(),
        run_subprocess=subprocess_mod.run,
    )


def build_windows_node_probe_deps(namespace: Mapping[str, Any]) -> WindowsNodeProbeDeps:
    json_mod = _ns(namespace, "json")
    os_mod = _ns(namespace, "os")
    queue_file = _ns(namespace, "QUEUE_FILE")

    def _queue_tasks() -> list[dict]:
        try:
            raw = json_mod.loads(queue_file.read_text())
            return list(raw.get("tasks", []))
        except Exception:
            return []

    return WindowsNodeProbeDeps(
        node_configs=_ns(namespace, "NODES"),
        queue_tasks=_queue_tasks,
        run_windows_ps=_ns(namespace, "_run_windows_ps"),
        run_subprocess=_ns(namespace, "subprocess").run,
        ssh_base_args=_ns(namespace, "_ssh_base_args"),
        windows_probe_error_hint=_ns(namespace, "_windows_probe_error_hint"),
        env_cpu_load_source=os_mod.environ.get("SCHEDULEURM_WINDOWS_CPU_LOAD_SOURCE"),
    )


def build_sudo_cpu_batch_probe_deps(namespace: Mapping[str, Any]) -> SudoCpuBatchProbeDeps:
    time_mod = _ns(namespace, "time")
    return SudoCpuBatchProbeDeps(
        node_configs=_ns(namespace, "NODES"),
        route_jump_cache=_ns(namespace, "_SSH_ROUTE_JUMP_CACHE"),
        outer_route_cache=_ns(namespace, "_SSH_OUTER_ROUTE_CACHE"),
        route_cache_ttl_s=_ns(namespace, "SSH_ROUTE_CACHE_TTL_S"),
        login_prefight_timeout_s=_ns(namespace, "SSH_LOGIN_PREFLIGHT_TIMEOUT_S"),
        now=time_mod.time,
        outer_ssh_route_key=_ns(namespace, "_outer_ssh_route_key"),
        configured_proxy_jumps=_ns(namespace, "_configured_proxy_jumps"),
        probe_ssh_proxy_jump=_ns(namespace, "_probe_ssh_proxy_jump"),
        remote_bash_command_for_node=_ns(namespace, "_remote_bash_command_for_node"),
        ssh_base_args_with_proxy=_ns(namespace, "_ssh_base_args_with_proxy"),
        run_ssh_subprocess=_ns(namespace, "_run_ssh_subprocess"),
    )


def build_probe_node_deps(namespace: Mapping[str, Any]) -> ProbeNodeDeps:
    time_mod = _ns(namespace, "time")
    return ProbeNodeDeps(
        node_configs=_ns(namespace, "NODES"),
        canonical_node_name=_ns(namespace, "_canonical_node_name"),
        node_is_windows=_ns(namespace, "_node_is_windows"),
        probe_windows_node=_ns(namespace, "_probe_windows_node"),
        path_exists=lambda path: Path(path).exists(),
        run_on=_ns(namespace, "run_on"),
        probe_windows_host_extras=_ns(namespace, "_probe_windows_host_extras"),
        sleep=time_mod.sleep,
    )


def build_probe_all_deps(namespace: Mapping[str, Any]) -> ProbeAllDeps:
    return ProbeAllDeps(
        node_configs=_ns(namespace, "NODES"),
        max_workers=_ns(namespace, "PROBE_ALL_MAX_WORKERS"),
        route_max_workers=_ns(namespace, "PROBE_ROUTE_MAX_WORKERS"),
        outer_ssh_route_key=_ns(namespace, "_outer_ssh_route_key"),
        probe_all_route_failures=_ns(namespace, "_probe_all_route_failures"),
        probe_node=_ns(namespace, "probe_node"),
        fold_claims_into_probe=_ns(namespace, "_fold_claims_into_probe"),
    )


def build_claim_probe_fold_deps(namespace: Mapping[str, Any]) -> ClaimProbeFoldDeps:
    claim_manager = _ns(namespace, "_ClaimManager")
    return ClaimProbeFoldDeps(
        claim_enabled_for=claim_manager.enabled_for,
        claim_snapshot=claim_manager.snapshot,
        notify=_ns(namespace, "notify"),
    )


def build_running_lifecycle_deps(namespace: Mapping[str, Any]) -> RunningLifecycleDeps:
    time_mod = _ns(namespace, "time")
    return RunningLifecycleDeps(
        set_current_usage=_ns(namespace, "_set_current_usage"),
        mark_probe_unknown=_ns(namespace, "_mark_probe_unknown"),
        clear_probe_unknown=_ns(namespace, "_clear_probe_unknown"),
        annotate_diag_after_unknown=_ns(namespace, "_annotate_diag_after_unknown"),
        cached_task_success_marker=_ns(namespace, "_cached_task_success_marker"),
        diagnose_terminal=_ns(namespace, "_diagnose_terminal"),
        release_task_claims_and_intents=_ns(namespace, "_release_task_claims_and_intents"),
        record_result_artifacts=_ns(namespace, "_record_result_artifacts"),
        requeue_after_crash=_ns(namespace, "_requeue_after_crash"),
        local_launch_transport_alive=_ns(namespace, "_local_launch_transport_alive"),
        terminal_diagnostic_matches=_ns(namespace, "_terminal_diagnostic_matches"),
        scan_completed_log_for_crash=_ns(namespace, "_scan_completed_log_for_crash"),
        mark_user_cancelled=_ns(namespace, "_mark_user_cancelled"),
        history_get=_ns(namespace, "history_get"),
        untrusted_startup_oom_sample=_ns(namespace, "_untrusted_startup_oom_sample"),
        history_record=_ns(namespace, "history_record"),
        runtime_history_record=_ns(namespace, "runtime_history_record"),
        apply_discovered_result_artifacts=_ns(namespace, "_apply_discovered_result_artifacts"),
        remember_last_placement=_ns(namespace, "_remember_last_placement"),
        declared_cpu_slot_floor=_ns(namespace, "_declared_cpu_slot_floor"),
        default_ram_mb=_ns(namespace, "DEFAULT_RAM_MB"),
        default_cpu_cores=_ns(namespace, "DEFAULT_CPU_CORES"),
        node_down_requeue_s=_ns(namespace, "NODE_DOWN_REQUEUE_S"),
        early_death_seconds=_ns(namespace, "EARLY_DEATH_SECONDS"),
        now=time_mod.time,
    )


def build_running_probe_snapshot_deps(namespace: Mapping[str, Any]) -> RunningProbeSnapshotDeps:
    return RunningProbeSnapshotDeps(
        state_lock=_ns(namespace, "state_lock"),
        load_state=_ns(namespace, "load_state"),
        running_probe_due=_ns(namespace, "_running_probe_due"),
        backend_batch_probe=_ns(namespace, "_BACKEND").batch_probe,
        notify=_ns(namespace, "notify"),
    )


def build_terminal_diagnostics_snapshot_deps(
    namespace: Mapping[str, Any],
) -> TerminalDiagnosticsSnapshotDeps:
    time_mod = _ns(namespace, "time")
    return TerminalDiagnosticsSnapshotDeps(
        state_lock=_ns(namespace, "state_lock"),
        load_state=_ns(namespace, "load_state"),
        scan_completed_log_for_crash=_ns(namespace, "_scan_completed_log_for_crash"),
        diagnose_terminal=_ns(namespace, "_diagnose_terminal"),
        discover_result_artifacts=_ns(namespace, "_discover_result_artifacts"),
        notify=_ns(namespace, "notify"),
        now=time_mod.time,
        max_tasks_per_cycle=_ns(namespace, "TERMINAL_DIAGNOSTICS_MAX_PER_CYCLE"),
        max_wall_s=_ns(namespace, "TERMINAL_DIAGNOSTICS_MAX_WALL_S"),
        include_log_result_artifacts=_ns(namespace, "TERMINAL_DIAGNOSTICS_INCLUDE_LOG_ARTIFACTS"),
        collect_terminal_evidence=_ns(namespace, "_collect_terminal_evidence"),
        node_down_requeue_s=_ns(namespace, "NODE_DOWN_REQUEUE_S"),
    )


def build_batch_check_running_deps(namespace: Mapping[str, Any]) -> BatchCheckRunningDeps:
    return BatchCheckRunningDeps(
        backend_batch_probe=_ns(namespace, "_BACKEND").batch_probe,
        running_lifecycle_deps=_ns(namespace, "_running_lifecycle_deps"),
        handle_unknown_probe_result=_ns(namespace, "_handle_unknown_probe_result"),
        clear_probe_unknown=_ns(namespace, "_clear_probe_unknown"),
        handle_dead_probe_result=_ns(namespace, "_handle_dead_probe_result"),
        apply_alive_probe_result=_ns(namespace, "_apply_alive_probe_result"),
    )


def build_update_running_tasks_deps(namespace: Mapping[str, Any]) -> UpdateRunningTasksDeps:
    time_mod = _ns(namespace, "time")
    return UpdateRunningTasksDeps(
        now=time_mod.time,
        running_probe_due=_ns(namespace, "_running_probe_due"),
        batch_check_running=_ns(namespace, "_batch_check_running"),
        eta_refresh_due=_ns(namespace, "_eta_refresh_due"),
        refresh_eta_from_logs=_ns(namespace, "_refresh_eta_from_logs"),
    )


def build_eta_tail_output_deps(namespace: Mapping[str, Any]) -> EtaTailOutputDeps:
    return EtaTailOutputDeps(
        node_is_windows=_ns(namespace, "_node_is_windows"),
        ps_quote=_ns(namespace, "_ps_quote"),
        windows_tail_ps=_ns(namespace, "_windows_tail_ps"),
        windows_path_for_task=_ns(namespace, "_windows_path_for_task"),
        run_windows_ps=_ns(namespace, "_run_windows_ps"),
        run_on=_ns(namespace, "run_on"),
        timeout_s=_ns(namespace, "ETA_TAIL_PROBE_TIMEOUT_S"),
    )


def build_eta_tail_snapshot_deps(namespace: Mapping[str, Any]) -> EtaTailSnapshotDeps:
    return EtaTailSnapshotDeps(
        state_lock=_ns(namespace, "state_lock"),
        load_state=_ns(namespace, "load_state"),
        eta_refresh_due=_ns(namespace, "_eta_refresh_due"),
        eta_tail_targets=_ns(namespace, "_eta_tail_targets"),
        eta_tail_outputs_by_node=_ns(namespace, "_eta_tail_outputs_by_node"),
        notify=_ns(namespace, "notify"),
    )


def build_eta_background_refresh_deps(
    namespace: Mapping[str, Any],
) -> EtaBackgroundRefreshDeps:
    time_mod = _ns(namespace, "time")
    return EtaBackgroundRefreshDeps(
        state_lock=_ns(namespace, "state_lock"),
        load_state=_ns(namespace, "load_state"),
        save_state=_ns(namespace, "save_state"),
        eta_refresh_due=_ns(namespace, "_eta_refresh_due"),
        eta_tail_targets=_ns(namespace, "_eta_tail_targets"),
        eta_tail_outputs_by_node=_ns(namespace, "_eta_tail_outputs_by_node"),
        refresh_eta_from_logs=_ns(namespace, "_refresh_eta_from_logs"),
        notify=_ns(namespace, "notify"),
        now=time_mod.time,
        record_eta_analysis_tasks=_ns(namespace, "_record_eta_analysis_tasks"),
    )


def build_eta_refresh_deps(namespace: Mapping[str, Any]) -> EtaRefreshDeps:
    time_mod = _ns(namespace, "time")
    return EtaRefreshDeps(
        load_eta_tracker_module=_ns(namespace, "_load_eta_tracker_module"),
        load_runtime_history=_ns(namespace, "_candidate_runtime_history"),
        runtime_history_closest_index=_ns(namespace, "_candidate_runtime_closest_index"),
        eta_tail_targets=_ns(namespace, "_eta_tail_targets"),
        history_get=_ns(namespace, "history_get"),
        runtime_total_history_s=_ns(namespace, "_runtime_total_history_s"),
        effective_elapsed_s=_ns(namespace, "_effective_elapsed_s"),
        history_fallback_eta_seconds=_ns(namespace, "_history_fallback_eta_seconds"),
        eta_tail_outputs_by_node=_ns(namespace, "_eta_tail_outputs_by_node"),
        last_progress_line=_ns(namespace, "_last_progress_line"),
        bapr_batch_projection=_ns(namespace, "_bapr_batch_projection"),
        apply_runtime_projection=_ns(namespace, "_apply_runtime_projection"),
        eta_confidence_for_source=_ns(namespace, "_eta_confidence_for_source"),
        now=time_mod.time,
    )
