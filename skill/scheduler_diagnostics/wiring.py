from __future__ import annotations

from typing import Any, Mapping

from scheduler_claim.runtime import ClaimManagerDeps
from scheduler_failure.crash_requeue import CrashRequeueDeps
from scheduler_diagnostics.doctor import DoctorDeps
from scheduler_runtime.history import ResourceHistoryDeps, RuntimeHistoryLookupDeps
from scheduler_probe.profile_local import ProfileLocalDeps
from scheduler_recovery.queued_artifact_reconcile import QueuedArtifactReconcileDeps
from scheduler_resource.forensics import ResourceForensicsDeps
from scheduler_result.discovery import ResultArtifactDeps
from scheduler_runtime.history_record import RuntimeHistoryRecordDeps
from scheduler_failure.terminal_diagnosis import TerminalDiagnosisDeps


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_claim_manager_deps(namespace: Mapping[str, Any]) -> ClaimManagerDeps:
    return ClaimManagerDeps(
        state_dir=_ns(namespace, "STATE_DIR"),
        node_configs=_ns(namespace, "NODES"),
        claim_ttl_s=_ns(namespace, "CLAIM_TTL_S"),
        claim_intent_ttl_s=_ns(namespace, "CLAIM_INTENT_TTL_S"),
        claim_fifo_strict_after_s=_ns(namespace, "CLAIM_FIFO_STRICT_AFTER_S"),
        claim_live_check=_ns(namespace, "CLAIM_LIVE_CHECK"),
        vram_margin_mb=_ns(namespace, "VRAM_MARGIN_MB"),
        one_third_pack_rule=_ns(namespace, "ONE_THIRD_PACK_RULE"),
        one_third_pack_grace_mb=_ns(namespace, "ONE_THIRD_PACK_GRACE_MB"),
        gpu_empty_used_mb=_ns(namespace, "GPU_EMPTY_USED_MB"),
        default_cpu_cores=_ns(namespace, "DEFAULT_CPU_CORES"),
        default_ram_mb=_ns(namespace, "DEFAULT_RAM_MB"),
        ignore_cpu_for_server_gpu_task=_ns(namespace, "_ignore_cpu_for_server_gpu_task"),
        task_ignores_one_third_pack_rule=_ns(namespace, "_task_ignores_one_third_pack_rule"),
        claims_remote_op=_ns(namespace, "_claims_remote_op"),
    )


def build_result_artifact_deps(namespace: Mapping[str, Any]) -> ResultArtifactDeps:
    time_mod = _ns(namespace, "time")
    return ResultArtifactDeps(
        node_configs=_ns(namespace, "NODES"),
        fetch_log_tail=_ns(namespace, "_fetch_log_tail"),
        node_is_windows=_ns(namespace, "_node_is_windows"),
        windows_path_for_task=_ns(namespace, "_windows_path_for_task"),
        ps_quote=_ns(namespace, "_ps_quote"),
        run_windows_ps=_ns(namespace, "_run_windows_ps"),
        run_on=_ns(namespace, "run_on"),
        run_subprocess=_ns(namespace, "subprocess").run,
        now=time_mod.time,
    )


def build_resource_history_deps(namespace: Mapping[str, Any]) -> ResourceHistoryDeps:
    time_mod = _ns(namespace, "time")
    return ResourceHistoryDeps(
        load_history=_ns(namespace, "load_history"),
        save_history=_ns(namespace, "save_history"),
        max_entries=_ns(namespace, "HISTORY_MAX_ENTRIES"),
        samples_per_sig=_ns(namespace, "HISTORY_SAMPLES_PER_SIG"),
        percentile_value=_ns(namespace, "HISTORY_PERCENTILE"),
        now=time_mod.time,
    )


def build_terminal_diagnosis_deps(namespace: Mapping[str, Any]) -> TerminalDiagnosisDeps:
    time_mod = _ns(namespace, "time")
    return TerminalDiagnosisDeps(
        node_configs=_ns(namespace, "NODES"),
        crash_patterns=_ns(namespace, "CRASH_PATTERNS"),
        training_markers=_ns(namespace, "TRAINING_MARKERS"),
        early_death_seconds=_ns(namespace, "EARLY_DEATH_SECONDS"),
        short_live_seconds=_ns(namespace, "SHORT_LIVE_SECONDS"),
        node_is_windows=_ns(namespace, "_node_is_windows"),
        fetch_windows_text_tail=_ns(namespace, "_fetch_windows_text_tail"),
        run_on=_ns(namespace, "run_on"),
        success_patterns_for_task=_ns(namespace, "_success_patterns_for_task"),
        cmd_looks_like_eval_or_benchmark=_ns(namespace, "_cmd_looks_like_eval_or_benchmark"),
        scan_full_log_for_success=_ns(namespace, "_scan_full_log_for_success"),
        terminal_final_model_success=_ns(namespace, "_terminal_final_model_success"),
        terminal_result_artifact_success=_ns(namespace, "_terminal_result_artifact_success"),
        now=time_mod.time,
    )


def build_crash_requeue_deps(namespace: Mapping[str, Any]) -> CrashRequeueDeps:
    time_mod = _ns(namespace, "time")
    claim_manager = _ns(namespace, "_ClaimManager")
    return CrashRequeueDeps(
        max_auto_retry=_ns(namespace, "MAX_AUTO_RETRY"),
        allocate_task_id=_ns(namespace, "_allocate_task_id"),
        recorded_launch_safety_state=_ns(namespace, "_recorded_launch_safety_state"),
        task_run_identity=_ns(namespace, "_task_run_identity"),
        has_user_cancelled_retry_descendant=_ns(namespace, "_has_user_cancelled_retry_descendant"),
        classify_failure=_ns(namespace, "_classify_failure"),
        write_escalation=_ns(namespace, "_write_escalation"),
        clear_live_eta_fields=_ns(namespace, "_clear_live_eta_fields"),
        local_user=_ns(namespace, "_local_user"),
        local_host_short=_ns(namespace, "_local_host_short"),
        scheduler_id=claim_manager.scheduler_id,
        now=time_mod.time,
    )


def build_resource_forensics_deps(namespace: Mapping[str, Any]) -> ResourceForensicsDeps:
    time_mod = _ns(namespace, "time")
    return ResourceForensicsDeps(
        now=time_mod.time,
        effective_elapsed_s=_ns(namespace, "_effective_elapsed_s"),
        discover_result_artifacts=_ns(namespace, "_discover_result_artifacts"),
        compact_resource_task_summary=_ns(namespace, "_compact_resource_task_summary"),
        gpu_threshold_snapshot=_ns(namespace, "_gpu_threshold_snapshot"),
        node_ram_snapshot=_ns(namespace, "_node_ram_snapshot"),
        last_progress_line=_ns(namespace, "_last_progress_line"),
        load_eta_tracker_module=_ns(namespace, "_load_eta_tracker_module"),
        ram_evict_ckpt_protect_progress=_ns(namespace, "RAM_EVICT_CKPT_PROTECT_PROGRESS"),
        ram_evict_ckpt_protect_min_age_s=_ns(namespace, "RAM_EVICT_CKPT_PROTECT_MIN_AGE_S"),
    )


def build_queued_artifact_reconcile_deps(namespace: Mapping[str, Any]) -> QueuedArtifactReconcileDeps:
    time_mod = _ns(namespace, "time")
    return QueuedArtifactReconcileDeps(
        backend=_ns(namespace, "_BACKEND"),
        task_pids=_ns(namespace, "_task_pids"),
        set_current_usage=_ns(namespace, "_set_current_usage"),
        release_task_claims_and_intents=_ns(namespace, "_release_task_claims_and_intents"),
        diagnose_terminal=_ns(namespace, "_diagnose_terminal"),
        requeue_after_crash=_ns(namespace, "_requeue_after_crash"),
        now=time_mod.time,
    )


def build_runtime_history_lookup_deps(namespace: Mapping[str, Any]) -> RuntimeHistoryLookupDeps:
    time_mod = _ns(namespace, "time")
    return RuntimeHistoryLookupDeps(
        load_runtime_history=_ns(namespace, "load_runtime_history"),
        history_get=_ns(namespace, "history_get"),
        task_runtime_keys=_ns(namespace, "_task_runtime_keys"),
        runtime_total_units_from_cmd=_ns(namespace, "_runtime_total_units_from_cmd"),
        queued_has_stale_live_eta=_ns(namespace, "_queued_has_stale_live_eta"),
        clear_live_eta_fields=_ns(namespace, "_clear_live_eta_fields"),
        eta_confidence_for_source=_ns(namespace, "_eta_confidence_for_source"),
        min_walltime_s=_ns(namespace, "RUNTIME_MIN_WALLTIME_S"),
        walltime_mult=_ns(namespace, "RUNTIME_WALLTIME_MULT"),
        closest_min_score=_ns(namespace, "RUNTIME_CLOSEST_MIN_SCORE"),
        now=time_mod.time,
    )


def build_runtime_history_record_deps(namespace: Mapping[str, Any]) -> RuntimeHistoryRecordDeps:
    time_mod = _ns(namespace, "time")
    return RuntimeHistoryRecordDeps(
        load_runtime_history=_ns(namespace, "load_runtime_history"),
        save_runtime_history=_ns(namespace, "save_runtime_history"),
        task_runtime_keys=_ns(namespace, "_task_runtime_keys"),
        runtime_device_kind_for_task=_ns(namespace, "_runtime_device_kind_for_task"),
        runtime_node_bucket_key=_ns(namespace, "_runtime_node_bucket_key"),
        percentile=_ns(namespace, "_percentile"),
        max_entries=_ns(namespace, "RUNTIME_HISTORY_MAX_ENTRIES"),
        samples_per_key=_ns(namespace, "RUNTIME_HISTORY_SAMPLES_PER_KEY"),
        percentile_value=_ns(namespace, "RUNTIME_HISTORY_PERCENTILE"),
        min_walltime_s=_ns(namespace, "RUNTIME_MIN_WALLTIME_S"),
        walltime_mult=_ns(namespace, "RUNTIME_WALLTIME_MULT"),
        now=time_mod.time,
    )


def build_doctor_deps(namespace: Mapping[str, Any]) -> DoctorDeps:
    return DoctorDeps(
        cmd_flag_values=_ns(namespace, "_cmd_flag_values"),
        arg_value=_ns(namespace, "_arg_value"),
        safe_read_text=_ns(namespace, "_safe_read_text"),
        queued_wait_for_file_block_reason=_ns(namespace, "_queued_wait_for_file_block_reason"),
        release_task_claims_and_intents=_ns(namespace, "_release_task_claims_and_intents"),
        queued_has_stale_live_eta=_ns(namespace, "_queued_has_stale_live_eta"),
        clear_live_eta_fields=_ns(namespace, "_clear_live_eta_fields"),
        seed_pending_eta_from_history=_ns(namespace, "_seed_pending_eta_from_history"),
        state_lock=_ns(namespace, "state_lock"),
        load_state=_ns(namespace, "load_state"),
        save_state=_ns(namespace, "save_state"),
    )


def build_profile_local_deps(namespace: Mapping[str, Any]) -> ProfileLocalDeps:
    time_mod = _ns(namespace, "time")
    return ProfileLocalDeps(
        log_dir=_ns(namespace, "LOG_DIR"),
        parse_env=_ns(namespace, "_parse_env"),
        project_from_path=_ns(namespace, "_project_from_path"),
        inject_python_u=_ns(namespace, "_inject_python_u"),
        local_backend_factory=_ns(namespace, "LocalBackend"),
        runtime_profile_from_log=_ns(namespace, "_runtime_profile_from_log"),
        apply_runtime_projection=_ns(namespace, "_apply_runtime_projection"),
        history_record=_ns(namespace, "history_record"),
        runtime_history_record=_ns(namespace, "runtime_history_record"),
        now=time_mod.time,
        sleep=time_mod.sleep,
    )
