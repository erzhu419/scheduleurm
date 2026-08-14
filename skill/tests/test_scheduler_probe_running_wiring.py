from __future__ import annotations

import json


def test_remote_and_probe_wiring_uses_scheduler_namespace(sch):
    remote = sch._remote_exec_deps()
    assert remote.node_configs is sch.NODES
    assert remote.canonical_node_name is sch._canonical_node_name
    assert remote.node_is_windows is sch._node_is_windows
    assert remote.raise_if_watcher_shutdown is sch._raise_if_watcher_shutdown
    assert remote.ssh_proxy_jump_cache is sch._SSH_PROXY_JUMP_CACHE
    assert remote.ssh_route_jump_cache is sch._SSH_ROUTE_JUMP_CACHE
    assert remote.ssh_outer_route_cache is sch._SSH_OUTER_ROUTE_CACHE
    assert remote.subprocess_run is sch.subprocess.run
    assert remote.subprocess_check_output is sch.subprocess.check_output
    assert remote.now is sch.time.time

    node = sch._probe_node_deps()
    assert node.node_configs is sch.NODES
    assert node.canonical_node_name is sch._canonical_node_name
    assert node.node_is_windows is sch._node_is_windows
    assert node.probe_windows_node is sch._probe_windows_node
    assert node.run_on is sch.run_on
    assert node.sleep is sch.time.sleep

    all_nodes = sch._probe_all_deps()
    assert all_nodes.node_configs is sch.NODES
    assert all_nodes.max_workers == sch.PROBE_ALL_MAX_WORKERS
    assert all_nodes.route_max_workers == sch.PROBE_ROUTE_MAX_WORKERS
    assert all_nodes.probe_node is sch.probe_node
    assert all_nodes.fold_claims_into_probe is sch._fold_claims_into_probe


def test_windows_and_sudo_probe_wiring_uses_scheduler_namespace(sch):
    sch.QUEUE_FILE.write_text(json.dumps({"tasks": [{"id": "t1"}, {"id": "t2"}]}))

    win = sch._windows_node_probe_deps()
    assert win.node_configs is sch.NODES
    assert win.queue_tasks() == [{"id": "t1"}, {"id": "t2"}]
    assert win.run_windows_ps is sch._run_windows_ps
    assert win.run_subprocess is sch.subprocess.run
    assert win.ssh_base_args is sch._ssh_base_args
    assert win.windows_probe_error_hint is sch._windows_probe_error_hint

    host = sch._windows_host_extras_deps()
    assert host.run_subprocess is sch.subprocess.run

    sudo = sch._sudo_cpu_batch_probe_deps()
    assert sudo.node_configs is sch.NODES
    assert sudo.route_jump_cache is sch._SSH_ROUTE_JUMP_CACHE
    assert sudo.outer_route_cache is sch._SSH_OUTER_ROUTE_CACHE
    assert sudo.now is sch.time.time
    assert sudo.remote_bash_command_for_node is sch._remote_bash_command_for_node
    assert sudo.run_ssh_subprocess is sch._run_ssh_subprocess


def test_running_and_eta_wiring_uses_scheduler_namespace(sch):
    lifecycle = sch._running_lifecycle_deps()
    assert lifecycle.set_current_usage is sch._set_current_usage
    assert lifecycle.mark_probe_unknown is sch._mark_probe_unknown
    assert lifecycle.diagnose_terminal is sch._diagnose_terminal
    assert lifecycle.requeue_after_crash is sch._requeue_after_crash
    assert lifecycle.history_record is sch.history_record
    assert lifecycle.runtime_history_record is sch.runtime_history_record
    assert lifecycle.default_ram_mb == sch.DEFAULT_RAM_MB
    assert lifecycle.default_cpu_cores == sch.DEFAULT_CPU_CORES
    assert lifecycle.node_down_requeue_s == sch.NODE_DOWN_REQUEUE_S

    snapshot = sch._running_probe_snapshot_deps()
    assert snapshot.state_lock is sch.state_lock
    assert snapshot.load_state is sch.load_state
    assert snapshot.running_probe_due is sch._running_probe_due
    assert snapshot.backend_batch_probe.__self__ is sch._BACKEND
    assert snapshot.notify is sch.notify

    batch = sch._batch_check_running_deps()
    assert batch.backend_batch_probe.__self__ is sch._BACKEND
    assert batch.running_lifecycle_deps is sch._running_lifecycle_deps
    assert batch.handle_dead_probe_result is sch._handle_dead_probe_result
    assert batch.apply_alive_probe_result is sch._apply_alive_probe_result

    update = sch._update_running_tasks_deps()
    assert update.now is sch.time.time
    assert update.batch_check_running is sch._batch_check_running
    assert update.refresh_eta_from_logs is sch._refresh_eta_from_logs

    eta_tail = sch._eta_tail_output_deps()
    assert eta_tail.node_is_windows is sch._node_is_windows
    assert eta_tail.windows_tail_ps is sch._windows_tail_ps
    assert eta_tail.run_windows_ps is sch._run_windows_ps
    assert eta_tail.run_on is sch.run_on

    eta = sch._eta_refresh_deps()
    assert eta.load_eta_tracker_module is sch._load_eta_tracker_module
    assert eta.load_runtime_history is sch._candidate_runtime_history
    assert eta.runtime_history_closest_index is sch._candidate_runtime_closest_index
    assert eta.history_get is sch.history_get
    assert eta.effective_elapsed_s is sch._effective_elapsed_s
    assert eta.bapr_batch_projection is sch._bapr_batch_projection
    assert eta.now is sch.time.time

    eta_background = sch._eta_background_refresh_deps()
    assert eta_background.state_lock is sch.state_lock
    assert eta_background.load_state is sch.load_state
    assert eta_background.save_state is sch.save_state
    assert eta_background.eta_refresh_due is sch._eta_refresh_due
    assert eta_background.record_eta_analysis_tasks is sch._record_eta_analysis_tasks
    assert eta_background.eta_tail_outputs_by_node is sch._eta_tail_outputs_by_node
    assert eta_background.refresh_eta_from_logs is sch._refresh_eta_from_logs


def test_running_update_wrapper_forwards_deferred_eta_refresh(sch, monkeypatch):
    calls = []

    def fake_update(state, **kwargs):
        calls.append((state, kwargs))

    monkeypatch.setattr(sch, "_update_running_tasks_impl", fake_update)

    state = {"tasks": []}
    sch.update_running_tasks(state, defer_eta_refresh=True)

    assert calls[0][0] is state
    assert calls[0][1]["defer_eta_refresh"] is True
