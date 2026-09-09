from __future__ import annotations

from typing import Any, Mapping

from .command import DispatchCommandDeps
from .core import DispatchAccountingDeps
from .launch_execution import DispatchLaunchExecutionDeps
from .launch_staging import DispatchLaunchStagingDeps
from .loop import DispatchLoopDeps
from .placement_apply import DispatchPlacementApplyDeps
from scheduler_migration.dispatch_resume_placement import DispatchResumePlacementDeps
from .task_gates import DispatchTaskGateDeps
from scheduler_resource.estimates import ResourceEstimateDeps
from scheduler_watch.command import WatchCommandDeps
from scheduler_watch.iteration import WatchIterationDeps
from scheduler_watch.state_phase import WatchStatePhaseDeps


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_dispatch_accounting_deps(namespace: Mapping[str, Any]) -> DispatchAccountingDeps:
    return DispatchAccountingDeps(
        counts_against_node_concurrency=_ns(namespace, "_counts_against_node_concurrency"),
        apply_cpu_slot_accounting_to_nodes=_ns(namespace, "_apply_cpu_slot_accounting_to_nodes"),
    )


def build_resource_estimate_deps(namespace: Mapping[str, Any]) -> ResourceEstimateDeps:
    time_mod = _ns(namespace, "time")
    return ResourceEstimateDeps(
        maybe_lower_explicit_resource_estimate=_ns(namespace, "_maybe_lower_explicit_resource_estimate"),
        effective_est_vram=_ns(namespace, "_effective_est_vram"),
        effective_est_ram=_ns(namespace, "_effective_est_ram"),
        live_sibling_ram_floor=_ns(namespace, "_live_sibling_ram_floor"),
        live_sibling_resource_estimate_index=_ns(namespace, "_live_sibling_resource_estimate_index"),
        description_sibling_resource_estimate_index=(
            _ns(namespace, "_description_sibling_resource_estimate_index")
        ),
        now=time_mod.time,
    )


def build_dispatch_launch_staging_deps(namespace: Mapping[str, Any]) -> DispatchLaunchStagingDeps:
    time_mod = _ns(namespace, "time")
    return DispatchLaunchStagingDeps(
        max_launch_retry=_ns(namespace, "MAX_LAUNCH_RETRY"),
        launch_max_cwd_size_mb=_ns(namespace, "LAUNCH_MAX_CWD_SIZE_MB"),
        stage_failure_reason=_ns(namespace, "_stage_failure_reason"),
        release_task_claims_and_intents=_ns(namespace, "_release_task_claims_and_intents"),
        write_escalation=_ns(namespace, "_write_escalation"),
        now=time_mod.time,
    )


def build_dispatch_launch_execution_deps(namespace: Mapping[str, Any]) -> DispatchLaunchExecutionDeps:
    time_mod = _ns(namespace, "time")
    return DispatchLaunchExecutionDeps(
        resource_snapshot_for_placement=_ns(namespace, "_resource_snapshot_for_placement"),
        new_launch_token=_ns(namespace, "_new_launch_token"),
        debit_node_resources_for_launch=_ns(namespace, "_debit_node_resources_for_launch"),
        task_run_identity=_ns(namespace, "_task_run_identity"),
        save_state=_ns(namespace, "save_state"),
        launch=_ns(namespace, "launch"),
        apply_launch_result_to_task=_ns(namespace, "_apply_launch_result_to_task"),
        notify=_ns(namespace, "notify"),
        now=time_mod.time,
    )


def build_dispatch_placement_apply_deps(namespace: Mapping[str, Any]) -> DispatchPlacementApplyDeps:
    return DispatchPlacementApplyDeps(
        node_configs=_ns(namespace, "NODES"),
        algorithm_name=_ns(namespace, "_algorithm_name"),
        algorithm_config_snapshot=_ns(namespace, "_algorithm_config_snapshot"),
        node_cpu_fallback_block_reason=_ns(namespace, "_node_cpu_fallback_block_reason"),
        task_cpu_fallback_capability=_ns(namespace, "_task_cpu_fallback_capability"),
        algorithm_selected_gpu_audit=_ns(namespace, "_algorithm_selected_gpu_audit"),
        assign_windows_pin_plan=_ns(namespace, "_assign_windows_pin_plan"),
    )


def build_dispatch_resume_placement_deps(namespace: Mapping[str, Any]) -> DispatchResumePlacementDeps:
    time_mod = _ns(namespace, "time")
    return DispatchResumePlacementDeps(
        task_requires_resume_scan=_ns(namespace, "_task_requires_resume_scan"),
        cached_resume_locations_for_task=_ns(namespace, "_cached_resume_locations_for_task"),
        allow_initial_resume_scan_error=_ns(namespace, "_allow_initial_resume_scan_error"),
        resume_checkpoint_stage_check=_ns(namespace, "_resume_checkpoint_stage_check"),
        record_staged_resume_location=_ns(namespace, "_record_staged_resume_location"),
        checkpoint_migration_extra_allowed_nodes=_ns(namespace, "_checkpoint_migration_extra_allowed_nodes"),
        soft_require_nodes=_ns(namespace, "_hpc_cpu_pool_soft_require_nodes"),
        release_task_claims_and_intents=_ns(namespace, "_release_task_claims_and_intents"),
        now=time_mod.time,
    )


def build_dispatch_task_gate_deps(namespace: Mapping[str, Any]) -> DispatchTaskGateDeps:
    return DispatchTaskGateDeps(
        clear_disallowed_cpu_fallback_selection=_ns(namespace, "_clear_disallowed_cpu_fallback_selection"),
        reconcile_queued_launch_artifacts_before_dispatch=_ns(
            namespace, "_reconcile_queued_launch_artifacts_before_dispatch"),
        task_run_identity=_ns(namespace, "_task_run_identity"),
        same_run_identity_live_artifact_reason=_ns(namespace, "_same_run_identity_live_artifact_reason"),
        eviction_cooldown_block_reason=_ns(namespace, "_eviction_cooldown_block_reason"),
        queued_wait_for_file_block_reason=_ns(namespace, "_queued_wait_for_file_block_reason"),
        queued_cpu_training_block_reason=_ns(namespace, "_queued_cpu_training_block_reason"),
    )


def build_dispatch_loop_deps(namespace: Mapping[str, Any]) -> DispatchLoopDeps:
    return DispatchLoopDeps(
        reconcile_requeue_lineage_invariants=_ns(namespace, "reconcile_requeue_lineage_invariants"),
        consider_migration=_ns(namespace, "_consider_migration"),
        preempt_for_high_priority=_ns(namespace, "_preempt_for_high_priority"),
        prepare_dispatch_node_accounting=_ns(namespace, "_prepare_dispatch_node_accounting"),
        dispatch_accounting_deps=_ns(namespace, "_dispatch_accounting_deps"),
        refresh_queued_resource_estimates=_ns(namespace, "_refresh_queued_resource_estimates"),
        load_history=_ns(namespace, "load_history"),
        resource_estimate_deps=_ns(namespace, "_resource_estimate_deps"),
        build_dispatch_queue_plan=_ns(namespace, "_build_dispatch_queue_plan"),
        task_run_identity=_ns(namespace, "_task_run_identity"),
        algorithm_global_batch_plan=_ns(namespace, "_algorithm_global_batch_plan"),
        apply_dispatch_task_gates=_ns(namespace, "_apply_dispatch_task_gates"),
        dispatch_task_gate_deps=_ns(namespace, "_dispatch_task_gate_deps"),
        resolve_resume_aware_placement=_ns(namespace, "_resolve_resume_aware_placement"),
        dispatch_resume_placement_deps=_ns(namespace, "_dispatch_resume_placement_deps"),
        build_no_fit_reason=_ns(namespace, "_build_no_fit_reason"),
        apply_dispatch_placement=_ns(namespace, "_apply_dispatch_placement"),
        dispatch_placement_apply_deps=_ns(namespace, "_dispatch_placement_apply_deps"),
        validate_selected_resume_checkpoint=_ns(namespace, "_validate_selected_resume_checkpoint"),
        release_task_claims_and_intents=_ns(namespace, "_release_task_claims_and_intents"),
        debit_node_resources_for_launch=_ns(namespace, "_debit_node_resources_for_launch"),
        precheck_git=_ns(namespace, "precheck_git"),
        resume_location_for_node=_ns(namespace, "_resume_location_for_node"),
        node_configs=_ns(namespace, "NODES"),
        stage_cwd_check=_ns(namespace, "_stage_cwd_check"),
        launch_input_stage_state=_ns(namespace, "_launch_input_stage_state"),
        apply_launch_staging_gate=_ns(namespace, "_apply_launch_staging_gate"),
        apply_launch_input_staging_gate=_ns(namespace, "_apply_launch_input_staging_gate"),
        dispatch_launch_staging_deps=_ns(namespace, "_dispatch_launch_staging_deps"),
        apply_dispatch_launch_execution=_ns(namespace, "_apply_dispatch_launch_execution"),
        dispatch_launch_execution_deps=_ns(namespace, "_dispatch_launch_execution_deps"),
        pick_placement=_ns(namespace, "pick_placement"),
    )


def build_watch_state_phase_deps(namespace: Mapping[str, Any]) -> WatchStatePhaseDeps:
    time_mod = _ns(namespace, "time")
    return WatchStatePhaseDeps(
        recover_queued_live_local_tasks=_ns(namespace, "recover_queued_live_local_tasks"),
        update_running_tasks=_ns(namespace, "update_running_tasks"),
        seed_pending_eta_from_history=_ns(namespace, "_seed_pending_eta_from_history"),
        detect_oom_kills_local=_ns(namespace, "_detect_oom_kills_local"),
        requeue_after_crash=_ns(namespace, "_requeue_after_crash"),
        notify=_ns(namespace, "notify"),
        classify_running_terminal_transitions=_ns(namespace, "_classify_running_terminal_transitions"),
        remember_running_resource_snapshots=_ns(namespace, "_remember_running_resource_snapshots"),
        reconcile_aggregate_only_vram=_ns(namespace, "_reconcile_aggregate_only_vram"),
        build_crash_forensics_payload=_ns(namespace, "_build_crash_forensics_payload"),
        build_oom_forensics_payload=_ns(namespace, "_build_oom_forensics_payload"),
        is_oom_like_forensics=_ns(namespace, "_is_oom_like_forensics"),
        enforce_post_dispatch_thresholds=_ns(namespace, "_enforce_post_dispatch_thresholds"),
        reserve_inflight_vram=_ns(namespace, "_reserve_inflight_vram"),
        do_dispatch=_ns(namespace, "_do_dispatch"),
        reconcile_external_tasks=_ns(namespace, "_reconcile_external_tasks"),
        archive_terminal_tasks=_ns(namespace, "archive_terminal_tasks"),
        detect_batch_completions=_ns(namespace, "_detect_batch_completions"),
        save_state=_ns(namespace, "save_state"),
        watch_dispatch_max_queued_per_cycle=_ns(namespace, "WATCH_DISPATCH_MAX_QUEUED_PER_CYCLE"),
        now=time_mod.time,
        recover_resolved_staging_failures=_ns(namespace, "_recover_resolved_staging_failures"),
    )


def build_dispatch_command_deps(namespace: Mapping[str, Any]) -> DispatchCommandDeps:
    return DispatchCommandDeps(
        set_hard_rule_mode=_ns(namespace, "_set_hard_rule_mode"),
        configure_algorithm=_ns(namespace, "_configure_algorithm"),
        state_lock=_ns(namespace, "state_lock"),
        load_state=_ns(namespace, "load_state"),
        save_state=_ns(namespace, "save_state"),
        recover_stale_launching_tasks_outside_lock=_ns(
            namespace, "recover_stale_launching_tasks_outside_lock"
        ),
        notify=_ns(namespace, "notify"),
        preload_docker_images_outside_lock=_ns(namespace, "_preload_docker_images_outside_lock"),
        stage_migration_candidates_outside_lock=_ns(namespace, "_stage_migration_candidates_outside_lock"),
        stage_launch_candidates_outside_lock=_ns(namespace, "_stage_launch_candidates_outside_lock"),
        sync_completed_results_outside_lock=_ns(namespace, "_request_result_sync_background"),
        running_probe_snapshot_outside_lock=_ns(namespace, "_running_probe_snapshot_outside_lock"),
        eta_tail_snapshot_outside_lock=_ns(namespace, "_eta_tail_snapshot_outside_lock"),
        probe_all=_ns(namespace, "probe_all"),
        probe_all_cached=_ns(namespace, "probe_all_cached"),
        prefetch_resume_scans_outside_lock=_ns(namespace, "_prefetch_resume_scans_outside_lock"),
        dispatch_probe_cache_max_age_s=_ns(namespace, "DISPATCH_PROBE_CACHE_MAX_AGE_S"),
        update_running_tasks=_ns(namespace, "update_running_tasks"),
        seed_pending_eta_from_history=_ns(namespace, "_seed_pending_eta_from_history"),
        remember_running_resource_snapshots=_ns(namespace, "_remember_running_resource_snapshots"),
        reconcile_aggregate_only_vram=_ns(namespace, "_reconcile_aggregate_only_vram"),
        reserve_inflight_vram=_ns(namespace, "_reserve_inflight_vram"),
        print_node_summary=_ns(namespace, "_print_node_summary"),
        do_dispatch=_ns(namespace, "_do_dispatch"),
        execute_deferred_launches=_ns(namespace, "_execute_deferred_launches"),
        load_state_shared_snapshot=_ns(namespace, "_load_state_shared_snapshot"),
        dispatch_cycle_log_payload=_ns(namespace, "_dispatch_cycle_log_payload"),
        format_task_location=_ns(namespace, "_format_task_location"),
        format_mem_gb=_ns(namespace, "_format_mem_gb"),
        max_launch_retry=_ns(namespace, "MAX_LAUNCH_RETRY"),
        defer_launch_outside_lock=_ns(namespace, "DEFER_LAUNCH_OUTSIDE_LOCK"),
        refresh_escalations_outside_lock=_ns(namespace, "_refresh_escalations_outside_lock"),
    )


def build_watch_command_deps(namespace: Mapping[str, Any]) -> WatchCommandDeps:
    os_mod = _ns(namespace, "os")
    signal_mod = _ns(namespace, "signal")
    time_mod = _ns(namespace, "time")
    watcher_state = _ns(namespace, "WATCHER_STATE")
    return WatchCommandDeps(
        resource_log_interval_s=_ns(namespace, "RESOURCE_LOG_INTERVAL_S"),
        reset_watcher_shutdown=_ns(namespace, "_reset_watcher_shutdown"),
        set_hard_rule_mode=_ns(namespace, "_set_hard_rule_mode"),
        configure_algorithm=_ns(namespace, "_configure_algorithm"),
        watcher_state_exists=lambda: watcher_state.exists(),
        read_watcher_state_text=lambda: watcher_state.read_text(),
        ensure_watcher_state_parent=lambda: watcher_state.parent.mkdir(parents=True, exist_ok=True),
        write_watcher_state_text=lambda text: watcher_state.write_text(text),
        unlink_watcher_state=lambda: watcher_state.unlink(),
        live_watcher_pid=_ns(namespace, "_live_watcher_pid"),
        pid_alive=_ns(namespace, "_pid_alive"),
        register_signal_handler=signal_mod.signal,
        sigterm=signal_mod.SIGTERM,
        sigint=signal_mod.SIGINT,
        request_watcher_shutdown=_ns(namespace, "_request_watcher_shutdown"),
        initial_watcher_state=_ns(namespace, "_initial_watcher_state"),
        getpid=os_mod.getpid,
        now=time_mod.time,
        notify=_ns(namespace, "notify"),
        smoke_test_envs_enabled=_ns(namespace, "_smoke_test_envs_enabled"),
        smoke_test_envs=_ns(namespace, "_smoke_test_envs"),
        state_lock=_ns(namespace, "state_lock"),
        load_state=_ns(namespace, "load_state"),
        save_state=_ns(namespace, "save_state"),
        recover_stale_launching_tasks_outside_lock=_ns(
            namespace, "recover_stale_launching_tasks_outside_lock"
        ),
        post_reboot_triage_announce=_ns(namespace, "_post_reboot_triage_announce"),
        watch_iteration=_ns(namespace, "_watch_iteration"),
        watcher_shutdown_requested_type=_ns(namespace, "_WatcherShutdownRequested"),
        sleep=time_mod.sleep,
        start_eta_periodic_refresh=_ns(namespace, "_start_eta_periodic_refresh"),
        stop_eta_periodic_refresh=_ns(namespace, "_stop_eta_periodic_refresh"),
    )


def build_watch_iteration_deps(namespace: Mapping[str, Any]) -> WatchIterationDeps:
    os_mod = _ns(namespace, "os")
    time_mod = _ns(namespace, "time")
    watcher_state = _ns(namespace, "WATCHER_STATE")
    return WatchIterationDeps(
        watch_respects_dispatch_intent=_ns(namespace, "WATCH_RESPECTS_DISPATCH_INTENT"),
        auto_adopt_interval_s=_ns(namespace, "AUTO_ADOPT_INTERVAL_S"),
        defer_launch_outside_lock=_ns(namespace, "DEFER_LAUNCH_OUTSIDE_LOCK"),
        archive_age_days=_ns(namespace, "ARCHIVE_AGE_DAYS"),
        resource_log_interval_s=_ns(namespace, "RESOURCE_LOG_INTERVAL_S"),
        state_lock=_ns(namespace, "state_lock"),
        load_state=_ns(namespace, "load_state"),
        save_state=_ns(namespace, "save_state"),
        notify=_ns(namespace, "notify"),
        read_dispatch_intent=_ns(namespace, "_read_dispatch_intent"),
        dispatch_intent_message=_ns(namespace, "_dispatch_intent_message"),
        recover_stale_launching_tasks_outside_lock=_ns(
            namespace, "recover_stale_launching_tasks_outside_lock"
        ),
        preload_docker_images_outside_lock=_ns(namespace, "_preload_docker_images_outside_lock"),
        stage_migration_candidates_outside_lock=_ns(namespace, "_stage_migration_candidates_outside_lock"),
        stage_launch_candidates_outside_lock=_ns(namespace, "_stage_launch_candidates_outside_lock"),
        stage_launch_candidates_background=lambda: _ns(
            namespace,
            "_stage_launch_candidates_outside_lock",
        )(background=True),
        probe_all=_ns(namespace, "probe_all"),
        probe_all_cached=_ns(namespace, "probe_all_cached"),
        watch_probe_cache_max_age_s=_ns(namespace, "WATCH_PROBE_CACHE_MAX_AGE_S"),
        watch_pre_dispatch_probe_max_age_s=_ns(
            namespace, "WATCH_PRE_DISPATCH_PROBE_MAX_AGE_S"
        ),
        prefetch_resume_scans_outside_lock=_ns(namespace, "_prefetch_resume_scans_outside_lock"),
        collect_external_task_probe_data=_ns(namespace, "_collect_external_task_probe_data"),
        running_probe_snapshot_outside_lock=_ns(namespace, "_running_probe_snapshot_outside_lock"),
        terminal_diagnostics_snapshot_outside_lock=_ns(namespace, "_terminal_diagnostics_snapshot_outside_lock"),
        eta_refresh_due=_ns(namespace, "_eta_refresh_due"),
        start_eta_refresh_background=_ns(namespace, "_start_eta_refresh_background"),
        record_eta_analysis_tasks=_ns(namespace, "_record_eta_analysis_tasks"),
        watch_phase_recorder_factory=_ns(namespace, "_WatchPhaseRecorder"),
        watch_phase_warn_s=_ns(namespace, "WATCH_PHASE_WARN_S"),
        run_watch_state_phase=_ns(namespace, "_run_watch_state_phase"),
        watch_state_phase_deps=_ns(namespace, "_watch_state_phase_deps"),
        execute_deferred_launches=_ns(namespace, "_execute_deferred_launches"),
        load_state_shared_snapshot=_ns(namespace, "_load_state_shared_snapshot"),
        notify_terminal_transitions=_ns(namespace, "_notify_terminal_transitions"),
        notify_dispatch_events=_ns(namespace, "_notify_dispatch_events"),
        compact_task_event_payload=_ns(namespace, "_compact_task_event_payload"),
        dispatch_cycle_log_payload=_ns(namespace, "_dispatch_cycle_log_payload"),
        sync_completed_results_outside_lock=_ns(namespace, "_request_result_sync_background"),
        tend_claims_for_watch=_ns(namespace, "_tend_claims_for_watch"),
        watcher_state_exists=lambda: watcher_state.exists(),
        read_watcher_state_text=lambda: watcher_state.read_text(),
        write_watcher_state_text=lambda text: watcher_state.write_text(text),
        plan_watcher_tick=_ns(namespace, "_plan_watcher_tick"),
        resource_accounting_payload=_ns(namespace, "_resource_accounting_payload"),
        build_heartbeat_payload=_ns(namespace, "_build_heartbeat_payload"),
        get_last_auto_adopt_at=_ns(namespace, "_get_last_auto_adopt_at"),
        set_last_auto_adopt_at=_ns(namespace, "_set_last_auto_adopt_at"),
        now=time_mod.time,
        getpid=os_mod.getpid,
        begin_history_batch=_ns(namespace, "_begin_history_batch"),
        end_history_batch=_ns(namespace, "_end_history_batch"),
        flush_pending_history_updates=_ns(namespace, "_flush_pending_history_updates"),
        refresh_escalations_outside_lock=_ns(namespace, "_refresh_escalations_outside_lock"),
    )
