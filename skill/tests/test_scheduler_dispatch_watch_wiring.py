from __future__ import annotations

from pathlib import Path

from skill.scheduler_dispatch_watch_wiring import (
    build_dispatch_command_deps,
    build_watch_command_deps,
    build_watch_iteration_deps,
)


def test_dispatch_loop_deps_are_built_from_scheduler_namespace(sch):
    deps = sch._dispatch_loop_deps()

    assert deps.reconcile_requeue_lineage_invariants is sch.reconcile_requeue_lineage_invariants
    assert deps.consider_migration is sch._consider_migration
    assert deps.preempt_for_high_priority is sch._preempt_for_high_priority
    assert deps.prepare_dispatch_node_accounting is sch._prepare_dispatch_node_accounting
    assert deps.dispatch_accounting_deps is sch._dispatch_accounting_deps
    assert deps.refresh_queued_resource_estimates is sch._refresh_queued_resource_estimates
    assert deps.load_history is sch.load_history
    assert deps.resource_estimate_deps is sch._resource_estimate_deps
    assert deps.build_dispatch_queue_plan is sch._build_dispatch_queue_plan
    assert deps.task_run_identity is sch._task_run_identity
    assert deps.algorithm_global_batch_plan is sch._algorithm_global_batch_plan
    assert deps.apply_dispatch_task_gates is sch._apply_dispatch_task_gates
    assert deps.dispatch_task_gate_deps is sch._dispatch_task_gate_deps
    assert deps.resolve_resume_aware_placement is sch._resolve_resume_aware_placement
    assert deps.dispatch_resume_placement_deps is sch._dispatch_resume_placement_deps
    assert deps.apply_dispatch_placement is sch._apply_dispatch_placement
    assert deps.dispatch_placement_apply_deps is sch._dispatch_placement_apply_deps
    assert deps.validate_selected_resume_checkpoint is sch._validate_selected_resume_checkpoint
    assert deps.release_task_claims_and_intents is sch._release_task_claims_and_intents
    assert deps.node_configs is sch.NODES
    assert deps.apply_dispatch_launch_execution is sch._apply_dispatch_launch_execution
    assert deps.dispatch_launch_execution_deps is sch._dispatch_launch_execution_deps
    assert deps.pick_placement is sch.pick_placement


def test_dispatch_subdeps_are_built_from_scheduler_namespace(sch):
    accounting = sch._dispatch_accounting_deps()
    assert accounting.counts_against_node_concurrency is sch._counts_against_node_concurrency
    assert accounting.apply_cpu_slot_accounting_to_nodes is sch._apply_cpu_slot_accounting_to_nodes

    estimate = sch._resource_estimate_deps()
    assert estimate.maybe_lower_explicit_resource_estimate is sch._maybe_lower_explicit_resource_estimate
    assert estimate.effective_est_vram is sch._effective_est_vram
    assert estimate.effective_est_ram is sch._effective_est_ram
    assert estimate.live_sibling_ram_floor is sch._live_sibling_ram_floor
    assert estimate.live_sibling_resource_estimate_index is sch._live_sibling_resource_estimate_index
    assert (
        estimate.description_sibling_resource_estimate_index
        is sch._description_sibling_resource_estimate_index
    )
    assert estimate.now is sch.time.time

    launch = sch._dispatch_launch_execution_deps()
    assert launch.resource_snapshot_for_placement is sch._resource_snapshot_for_placement
    assert launch.new_launch_token is sch._new_launch_token
    assert launch.save_state is sch.save_state
    assert launch.launch is sch.launch
    assert launch.apply_launch_result_to_task is sch._apply_launch_result_to_task
    assert launch.notify is sch.notify

    placement = sch._dispatch_placement_apply_deps()
    assert placement.node_configs is sch.NODES
    assert placement.algorithm_name is sch._algorithm_name
    assert placement.algorithm_config_snapshot is sch._algorithm_config_snapshot
    assert placement.node_cpu_fallback_block_reason is sch._node_cpu_fallback_block_reason
    assert placement.algorithm_selected_gpu_audit is sch._algorithm_selected_gpu_audit

    gate = sch._dispatch_task_gate_deps()
    assert gate.clear_disallowed_cpu_fallback_selection is sch._clear_disallowed_cpu_fallback_selection
    assert gate.task_run_identity is sch._task_run_identity
    assert gate.same_run_identity_live_artifact_reason is sch._same_run_identity_live_artifact_reason
    assert gate.queued_wait_for_file_block_reason is sch._queued_wait_for_file_block_reason


def test_dispatch_and_watch_command_deps_are_built_from_scheduler_namespace(sch, tmp_path, monkeypatch):
    watcher_state = Path(tmp_path) / ".watcher_state.json"
    monkeypatch.setattr(sch, "WATCHER_STATE", watcher_state, raising=False)

    dispatch = build_dispatch_command_deps(vars(sch))
    assert dispatch.set_hard_rule_mode is sch._set_hard_rule_mode
    assert dispatch.configure_algorithm is sch._configure_algorithm
    assert dispatch.state_lock is sch.state_lock
    assert dispatch.load_state is sch.load_state
    assert dispatch.save_state is sch.save_state
    assert dispatch.probe_all is sch.probe_all
    assert dispatch.probe_all_cached is sch.probe_all_cached
    assert dispatch.prefetch_resume_scans_outside_lock is sch._prefetch_resume_scans_outside_lock
    assert dispatch.dispatch_probe_cache_max_age_s == sch.DISPATCH_PROBE_CACHE_MAX_AGE_S
    assert dispatch.update_running_tasks is sch.update_running_tasks
    assert dispatch.do_dispatch is sch._do_dispatch
    assert dispatch.execute_deferred_launches is sch._execute_deferred_launches
    assert dispatch.max_launch_retry == sch.MAX_LAUNCH_RETRY
    assert dispatch.defer_launch_outside_lock == sch.DEFER_LAUNCH_OUTSIDE_LOCK
    assert dispatch.refresh_escalations_outside_lock is sch._refresh_escalations_outside_lock

    watch_cmd = build_watch_command_deps(vars(sch))
    assert watch_cmd.resource_log_interval_s == sch.RESOURCE_LOG_INTERVAL_S
    assert watch_cmd.set_hard_rule_mode is sch._set_hard_rule_mode
    assert watch_cmd.configure_algorithm is sch._configure_algorithm
    assert watch_cmd.getpid is sch.os.getpid
    assert watch_cmd.now is sch.time.time
    assert watch_cmd.sleep is sch.time.sleep
    assert watch_cmd.watch_iteration is sch._watch_iteration
    assert watch_cmd.start_eta_periodic_refresh is sch._start_eta_periodic_refresh
    assert watch_cmd.stop_eta_periodic_refresh is sch._stop_eta_periodic_refresh
    assert watch_cmd.watcher_shutdown_requested_type is sch._WatcherShutdownRequested
    assert watch_cmd.watcher_state_exists() is False
    watch_cmd.ensure_watcher_state_parent()
    watch_cmd.write_watcher_state_text("{}")
    assert watch_cmd.watcher_state_exists() is True
    assert watch_cmd.read_watcher_state_text() == "{}"
    watch_cmd.unlink_watcher_state()
    assert watch_cmd.watcher_state_exists() is False


def test_watch_iteration_deps_are_built_from_scheduler_namespace(sch, tmp_path, monkeypatch):
    watcher_state = Path(tmp_path) / ".watcher_state.json"
    monkeypatch.setattr(sch, "WATCHER_STATE", watcher_state, raising=False)

    deps = build_watch_iteration_deps(vars(sch))
    assert deps.watch_respects_dispatch_intent == sch.WATCH_RESPECTS_DISPATCH_INTENT
    assert deps.auto_adopt_interval_s == sch.AUTO_ADOPT_INTERVAL_S
    assert deps.defer_launch_outside_lock == sch.DEFER_LAUNCH_OUTSIDE_LOCK
    assert deps.archive_age_days == sch.ARCHIVE_AGE_DAYS
    assert deps.resource_log_interval_s == sch.RESOURCE_LOG_INTERVAL_S
    assert deps.state_lock is sch.state_lock
    assert deps.load_state is sch.load_state
    assert deps.save_state is sch.save_state
    assert deps.notify is sch.notify
    assert deps.probe_all is sch.probe_all
    assert deps.eta_refresh_due is sch._eta_refresh_due
    assert deps.start_eta_refresh_background is sch._start_eta_refresh_background
    assert deps.record_eta_analysis_tasks is sch._record_eta_analysis_tasks
    assert deps.prefetch_resume_scans_outside_lock is sch._prefetch_resume_scans_outside_lock
    assert deps.begin_history_batch is sch._begin_history_batch
    assert deps.end_history_batch is sch._end_history_batch
    assert deps.flush_pending_history_updates is sch._flush_pending_history_updates
    assert deps.refresh_escalations_outside_lock is sch._refresh_escalations_outside_lock
    assert callable(deps.stage_launch_candidates_background)
    assert deps.watch_phase_recorder_factory is sch._WatchPhaseRecorder
    assert deps.watch_state_phase_deps is sch._watch_state_phase_deps
    assert deps.execute_deferred_launches is sch._execute_deferred_launches
    assert deps.dispatch_cycle_log_payload is sch._dispatch_cycle_log_payload
    assert deps.sync_completed_results_outside_lock is sch._request_result_sync_background
    assert deps.tend_claims_for_watch is sch._tend_claims_for_watch
    assert deps.get_last_auto_adopt_at is sch._get_last_auto_adopt_at
    assert deps.set_last_auto_adopt_at is sch._set_last_auto_adopt_at
    assert deps.now is sch.time.time
    assert deps.getpid is sch.os.getpid
    assert (
        deps.watch_pre_dispatch_probe_max_age_s
        == sch.WATCH_PRE_DISPATCH_PROBE_MAX_AGE_S
    )
    assert deps.watcher_state_exists() is False
    deps.write_watcher_state_text("{}")
    assert deps.watcher_state_exists() is True
    assert deps.read_watcher_state_text() == "{}"
