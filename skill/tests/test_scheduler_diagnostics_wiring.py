from __future__ import annotations


def _same_bound_method(left, right) -> bool:
    return (
        getattr(left, "__self__", None) is getattr(right, "__self__", None)
        and getattr(left, "__func__", None) is getattr(right, "__func__", None)
    )


def test_claim_result_and_history_wiring_uses_scheduler_namespace(sch):
    claim = sch._claim_manager_deps()
    assert claim.state_dir is sch.STATE_DIR
    assert claim.node_configs is sch.NODES
    assert claim.claim_ttl_s == sch.CLAIM_TTL_S
    assert claim.vram_margin_mb == sch.VRAM_MARGIN_MB
    assert claim.gpu_empty_used_mb == sch.GPU_EMPTY_USED_MB
    assert claim.ignore_cpu_for_server_gpu_task is sch._ignore_cpu_for_server_gpu_task
    assert claim.claims_remote_op is sch._claims_remote_op

    result = sch._result_artifact_deps()
    assert result.node_configs is sch.NODES
    assert result.fetch_log_tail is sch._fetch_log_tail
    assert result.node_is_windows is sch._node_is_windows
    assert result.run_subprocess is sch.subprocess.run
    assert result.now is sch.time.time

    history = sch._resource_history_deps()
    assert history.load_history is sch.load_history
    assert history.save_history is sch.save_history
    assert history.max_entries == sch.HISTORY_MAX_ENTRIES
    assert history.samples_per_sig == sch.HISTORY_SAMPLES_PER_SIG
    assert history.now is sch.time.time


def test_diagnosis_crash_and_forensics_wiring_uses_scheduler_namespace(sch):
    diagnosis = sch._terminal_diagnosis_deps()
    assert diagnosis.node_configs is sch.NODES
    assert diagnosis.crash_patterns is sch.CRASH_PATTERNS
    assert diagnosis.training_markers is sch.TRAINING_MARKERS
    assert diagnosis.fetch_windows_text_tail is sch._fetch_windows_text_tail
    assert diagnosis.terminal_final_model_success is sch._terminal_final_model_success
    assert diagnosis.now is sch.time.time

    crash = sch._crash_requeue_deps()
    assert crash.max_auto_retry == sch.MAX_AUTO_RETRY
    assert crash.allocate_task_id is sch._allocate_task_id
    assert crash.recorded_launch_safety_state is sch._recorded_launch_safety_state
    assert crash.clear_live_eta_fields is sch._clear_live_eta_fields
    assert _same_bound_method(crash.scheduler_id, sch._ClaimManager.scheduler_id)
    assert crash.now is sch.time.time

    forensics = sch._resource_forensics_deps()
    assert forensics.effective_elapsed_s is sch._effective_elapsed_s
    assert forensics.discover_result_artifacts is sch._discover_result_artifacts
    assert forensics.gpu_threshold_snapshot is sch._gpu_threshold_snapshot
    assert forensics.node_ram_snapshot is sch._node_ram_snapshot
    assert forensics.load_eta_tracker_module is sch._load_eta_tracker_module
    assert forensics.now is sch.time.time

    queued = sch._queued_artifact_reconcile_deps()
    assert queued.backend is sch._BACKEND
    assert queued.task_pids is sch._task_pids
    assert queued.diagnose_terminal is sch._diagnose_terminal
    assert queued.requeue_after_crash is sch._requeue_after_crash


def test_runtime_doctor_and_profile_wiring_uses_scheduler_namespace(sch):
    lookup = sch._runtime_history_lookup_deps()
    assert lookup.load_runtime_history is sch.load_runtime_history
    assert lookup.history_get is sch.history_get
    assert lookup.task_runtime_keys is sch._task_runtime_keys
    assert lookup.runtime_total_units_from_cmd is sch._runtime_total_units_from_cmd
    assert lookup.min_walltime_s == sch.RUNTIME_MIN_WALLTIME_S
    assert lookup.now is sch.time.time

    record = sch._runtime_history_record_deps()
    assert record.load_runtime_history is sch.load_runtime_history
    assert record.save_runtime_history is sch.save_runtime_history
    assert record.runtime_device_kind_for_task is sch._runtime_device_kind_for_task
    assert record.percentile is sch._percentile
    assert record.samples_per_key == sch.RUNTIME_HISTORY_SAMPLES_PER_KEY
    assert record.now is sch.time.time

    doctor = sch._doctor_deps()
    assert doctor.cmd_flag_values is sch._cmd_flag_values
    assert doctor.arg_value is sch._arg_value
    assert doctor.queued_wait_for_file_block_reason is sch._queued_wait_for_file_block_reason
    assert doctor.seed_pending_eta_from_history is sch._seed_pending_eta_from_history
    assert doctor.state_lock is sch.state_lock

    profile = sch._profile_local_deps()
    assert profile.log_dir is sch.LOG_DIR
    assert profile.parse_env is sch._parse_env
    assert profile.local_backend_factory is sch.LocalBackend
    assert profile.runtime_profile_from_log is sch._runtime_profile_from_log
    assert profile.runtime_history_record is sch.runtime_history_record
    assert profile.sleep is sch.time.sleep
