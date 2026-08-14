from __future__ import annotations

import builtins


def _same_bound_method(left, right) -> bool:
    return (
        getattr(left, "__self__", None) is getattr(right, "__self__", None)
        and getattr(left, "__func__", None) is getattr(right, "__func__", None)
    )


def test_placement_and_resume_wiring_uses_scheduler_namespace(sch):
    gate = sch._node_resource_gate_deps()
    assert gate.default_cpu_cores == sch.DEFAULT_CPU_CORES
    assert gate.hard_rule_bypassed is sch._hard_rule_bypassed
    assert gate.node_ram_headroom_mb is sch._node_ram_headroom_mb

    placement = sch._placement_runtime()
    assert placement.nodes_by_name is sch.NODES
    assert placement.vram_margin_mb == sch.VRAM_MARGIN_MB
    assert placement.hpc_cpu_pool_soft_require_nodes is sch._hpc_cpu_pool_soft_require_nodes
    assert placement.node_resources_ok is sch._node_resources_ok
    assert placement.algorithm_gpu_score is sch._algorithm_gpu_score
    assert placement.algorithm_gpu_fit_block_reason is sch._algorithm_gpu_fit_block_reason

    resume = sch._resume_scan_deps()
    assert resume.node_configs is sch.NODES
    assert resume.ckpt_exts == sch.CKPT_EXTS
    assert resume.resume_safe_name_re is sch.RESUME_SAFE_NAME_RE
    assert resume.run_on is sch.run_on
    assert resume.now is sch.time.time

    docker = sch._docker_wrap_deps()
    assert docker.env_deploy is sch.env_deploy
    assert docker.node_configs is sch.NODES
    assert docker.run_on is sch.run_on


def test_launch_staging_wiring_uses_scheduler_namespace(sch):
    cwd = sch._launch_cwd_staging_deps()
    assert cwd.node_configs is sch.NODES
    assert cwd.launch_max_cwd_size_mb == sch.LAUNCH_MAX_CWD_SIZE_MB
    assert cwd.staging_cache_hit is sch._staging_cache_hit
    assert cwd.stage_code_tar_for_launch is sch._stage_code_tar_for_launch
    assert cwd.relay_node_for_node is sch._relay_node_for_node
    assert cwd.run_control_subprocess is sch._run_control_subprocess

    ckpt = sch._resume_ckpt_staging_deps()
    assert ckpt.task_requires_resume_scan is sch._task_requires_resume_scan
    assert ckpt.resume_checkpoint_stage_check is sch._resume_checkpoint_stage_check
    assert ckpt.rsync_shell_for_pair is sch._rsync_shell_for_pair
    assert ckpt.run_subprocess is sch._run_control_subprocess

    mig = sch._migration_staging_deps()
    assert mig.migration_max_cwd_size_mb == sch.MIGRATION_MAX_CWD_SIZE_MB
    assert mig.apply_node_cmd_rewrites is sch._apply_node_cmd_rewrites
    assert mig.run_subprocess is sch._run_control_subprocess

    plan = sch._launch_staging_plan_deps()
    assert plan.node_configs is sch.NODES
    assert plan.queued_wait_for_file_block_reason is sch._queued_wait_for_file_block_reason
    assert plan.resume_checkpoint_stage_check is sch._resume_checkpoint_stage_check
    key = ("test-cap",)
    sch._STAGING_CAP_EXCEEDED[key] = sch.time.time()
    try:
        assert plan.staging_cap_recent(key) is True
    finally:
        sch._STAGING_CAP_EXCEEDED.pop(key, None)


def test_control_subprocess_preserves_patched_subprocess_run_hook(sch, monkeypatch):
    calls = []

    class Result:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_run(args, **kwargs):
        calls.append((list(args), kwargs))
        return Result()

    monkeypatch.setattr(sch.subprocess, "run", fake_run)

    result = sch._run_control_subprocess(
        ["rsync", "src/", "dst/"],
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0
    assert calls == [(
        ["rsync", "src/", "dst/"],
        {
            "timeout": 600,
            "input": None,
            "capture_output": True,
            "stdout": None,
            "stderr": None,
            "text": True,
        },
    )]


def test_migration_result_and_pressure_wiring_uses_scheduler_namespace(sch):
    resume_fit = sch._resume_fit_deps()
    assert resume_fit.node_configs is sch.NODES
    assert resume_fit.staging_cap_exceeded is sch._STAGING_CAP_EXCEEDED
    assert resume_fit.gpu_fits is sch._gpu_fits
    assert resume_fit.pick_placement is sch.pick_placement
    assert resume_fit.now is sch.time.time

    policy = sch._migration_policy_deps()
    assert policy.compute_node_load_seconds is sch.compute_node_load_seconds
    assert policy.blocked_nodes_for_task is sch._blocked_nodes_for_task
    assert policy.can_migrate_to is sch._can_migrate_to
    assert policy.migration_max_per_dispatch == sch.MIGRATION_MAX_PER_DISPATCH

    result = sch._result_sync_executor_deps()
    assert result.node_is_windows is sch._node_is_windows
    assert result.sync_windows_result is sch._sync_windows_result
    assert result.rsync_path_for_node is sch._rsync_path_for_node
    assert result.result_sync_timeout_s == sch.RESULT_SYNC_TIMEOUT_S

    post = sch._post_dispatch_threshold_deps()
    assert post.node_configs is sch.NODES
    assert post.effective_elapsed_s is sch._effective_elapsed_s
    assert post.task_evict_loss_protected is sch._task_evict_loss_protected
    assert post.evict_to_queue is sch._evict_to_queue
    assert post.now is sch.time.time


def test_eviction_and_recovery_wiring_uses_scheduler_namespace(sch):
    preempt = sch._preemption_deps()
    assert preempt.node_configs is sch.NODES
    assert preempt.queue_wait_min == sch.PREEMPT_QUEUE_WAIT_MIN
    assert preempt.evict_to_queue is sch._evict_to_queue
    assert preempt.format_mem_gb is sch._format_mem_gb

    eviction = sch._eviction_deps()
    assert eviction.release_task_claims_and_intents is sch._release_task_claims_and_intents
    assert eviction.kill_task_processes is sch._kill_task_processes
    assert eviction.local_cpu_evict_node_cooldown_s == sch.LOCAL_CPU_EVICT_NODE_COOLDOWN_S
    assert eviction.now is sch.time.time

    commit = sch._launch_commit_deps()
    assert commit.state_lock is sch.state_lock
    assert commit.apply_launch_result_to_task is sch._apply_launch_result_to_task
    assert _same_bound_method(commit.kill_task, sch._BACKEND.kill)

    orphan = sch._local_orphan_recovery_deps()
    assert orphan.state_dir is sch.STATE_DIR
    assert orphan.remember_last_placement is sch._remember_last_placement
    assert _same_bound_method(orphan.claim_enabled_for, sch._ClaimManager.enabled_for)
    assert _same_bound_method(orphan.update_claim_pid, sch._ClaimManager.update_pid)
    assert orphan.requires_local_capacity_check is sch._requires_local_capacity_check
    assert orphan.recover_remote_nodes == sch.RECOVER_QUEUED_LIVE_REMOTE_NODES

    terminal = sch._terminal_finalize_deps()
    assert terminal.state_dir is sch.STATE_DIR
    assert terminal.diagnose_terminal is sch._diagnose_terminal
    assert terminal.requeue_after_crash is sch._requeue_after_crash
    assert terminal.is_local_node("local") is True

    recovery = sch._launch_recovery_deps()
    assert recovery.try_recover_orphan_local_task is sch._try_recover_orphan_local_task
    assert recovery.try_finalize_terminal_local_task is sch._try_finalize_terminal_local_task
    assert recovery.clear_live_eta_fields is sch._clear_live_eta_fields


def test_watcher_support_wiring_uses_scheduler_namespace(sch):
    wait = sch._wait_for_deps()
    assert wait.state_lock is sch.state_lock
    assert wait.update_running_tasks is sch.update_running_tasks
    assert wait.running_probe_snapshot_outside_lock is sch._running_probe_snapshot_outside_lock
    assert wait.sleep is sch.time.sleep
    assert wait.print_fn is builtins.print

    notify = sch._notify_deps()
    assert notify.watcher_log is sch.WATCHER_LOG
    assert notify.feishu_config is sch.FEISHU_CONFIG
    assert notify.format_task_location is sch._format_task_location
    assert notify.now is sch.time.time

    adopt = sch._adopt_probe_deps()
    assert adopt.node_configs is sch.NODES
    assert adopt.node_is_windows is sch._node_is_windows
    assert adopt.descendants_cap == sch._DESCENDANTS_CAP

    reconcile = sch._external_reconcile_deps()
    assert reconcile.collect_external_task_probe_data is sch._collect_external_task_probe_data
    assert reconcile.is_scheduler_control_process is sch._is_scheduler_control_process
    assert reconcile.allocate_task_id is sch._allocate_task_id
    assert reconcile.default_ram_mb == sch.DEFAULT_RAM_MB

    accounting = sch._resource_accounting_deps()
    assert _same_bound_method(accounting.scheduler_id, sch._ClaimManager.scheduler_id)
    assert accounting.reserved_cpu_slots_for_task is sch._reserved_cpu_slots_for_task
    assert accounting.algorithm_name is sch._algorithm_name

    smoke = sch._env_smoke_deps()
    assert smoke.load_state is sch.load_state
    assert smoke.notify is sch.notify
    assert smoke.environ is sch.os.environ

    claims = sch._claim_tending_deps()
    assert _same_bound_method(claims.enabled_for_node, sch._ClaimManager.enabled_for)
    assert claims.collect_claim_renewal_inputs is sch._collect_claim_renewal_inputs
    assert _same_bound_method(claims.scheduler_id, sch._ClaimManager.scheduler_id)
    assert claims.notify is sch.notify
