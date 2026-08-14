from contextlib import contextmanager

from skill.scheduler_running.snapshots import (
    BatchCheckRunningDeps,
    EtaTailOutputDeps,
    RunningProbeSnapshotDeps,
    TerminalDiagnosticsSnapshotDeps,
    UpdateRunningTasksDeps,
    batch_check_running,
    effective_elapsed_s,
    eta_tail_outputs_by_node,
    running_probe_snapshot_outside_lock,
    terminal_diagnostic_fingerprint,
    terminal_diagnostics_snapshot_outside_lock,
    update_running_tasks,
)


@contextmanager
def _lock(*args, **kwargs):
    yield


def test_running_probe_snapshot_reads_under_lock_and_probes_after_unlock():
    events = []
    state = {"tasks": [{"id": "t1", "status": "running"}]}

    @contextmanager
    def lock(*args, **kwargs):
        events.append(("lock", kwargs.get("purpose")))
        yield
        events.append(("unlock", kwargs.get("purpose")))

    def backend_probe(snapshot):
        events.append(("probe", list(snapshot["tasks"])))
        return {"t1": {"state": "alive"}}

    result, ids = running_probe_snapshot_outside_lock(
        deps=RunningProbeSnapshotDeps(
            state_lock=lock,
            load_state=lambda: state,
            running_probe_due=lambda snapshot: True,
            backend_batch_probe=backend_probe,
            notify=lambda *a, **k: events.append(("notify", a)),
        )
    )

    assert result == {"t1": {"state": "alive"}}
    assert ids == {"t1"}
    assert [event[0] for event in events] == ["lock", "unlock", "probe"]


def test_running_probe_snapshot_retries_pending_terminal_without_full_probe():
    state = {
        "tasks": [
            {
                "id": "pending",
                "status": "running",
                "terminal_transition_deferred": True,
            },
            {"id": "active", "status": "running"},
        ]
    }

    result, ids = running_probe_snapshot_outside_lock(
        deps=RunningProbeSnapshotDeps(
            state_lock=_lock,
            load_state=lambda: state,
            running_probe_due=lambda snapshot: False,
            backend_batch_probe=lambda snapshot: (_ for _ in ()).throw(
                AssertionError("full backend probe should not run")
            ),
            notify=lambda *a, **k: None,
        )
    )

    assert ids == {"pending"}
    assert result["pending"]["state"] == "dead"
    assert result["pending"]["terminal_diagnosis_deferred"] is True
    assert result["pending"]["terminal_retry_only"] is True


def test_terminal_diagnostics_snapshot_precomputes_dead_tasks_only():
    running = {
        "id": "tdead",
        "status": "running",
        "node": "node001",
        "log_path": "/tmp/tdead.log",
        "started_at": 123.0,
        "cmd": "python train.py",
    }
    state = {
        "tasks": [
            running,
            {"id": "talive", "status": "running", "node": "node001"},
            {"id": "tdone", "status": "done", "node": "node001"},
        ]
    }
    artifacts_calls = []

    out = terminal_diagnostics_snapshot_outside_lock(
        {"tdead": {"state": "dead"}, "talive": {"state": "alive"}},
        {"tdead", "talive"},
        deps=TerminalDiagnosticsSnapshotDeps(
            state_lock=_lock,
            load_state=lambda: state,
            scan_completed_log_for_crash=lambda task: (False, ""),
            diagnose_terminal=lambda task: {"is_crash": False, "reason": "done"},
            discover_result_artifacts=lambda task, **kwargs: artifacts_calls.append((task["id"], kwargs)) or [],
            notify=lambda *a, **k: None,
            now=lambda: 456.0,
        ),
    )

    assert set(out) == {"tdead"}
    assert out["tdead"]["fingerprint"] == terminal_diagnostic_fingerprint(running)
    assert out["tdead"]["diagnosis"]["reason"] == "done"
    assert artifacts_calls == [("tdead", {"include_log": True})]


def test_terminal_diagnostics_snapshot_precomputes_due_unknown_reroutes():
    due = {
        "id": "tdue",
        "status": "running",
        "node": "node001",
        "log_path": "/tmp/tdue.log",
        "started_at": 123.0,
        "cmd": "python train.py",
        "reroute_on_node_down": True,
        "probe_unknown_since": 100.0,
    }
    launch_alive = {
        **due,
        "id": "talive-launch",
        "log_path": "/tmp/talive-launch.log",
    }
    recent = {
        **due,
        "id": "trecent",
        "log_path": "/tmp/trecent.log",
        "probe_unknown_since": 499.5,
    }
    state = {"tasks": [due, launch_alive, recent]}

    out = terminal_diagnostics_snapshot_outside_lock(
        {
            "tdue": {"state": "unknown", "launch_safety_state": "dead"},
            "talive-launch": {
                "state": "unknown",
                "launch_safety_state": "alive",
            },
            "trecent": {"state": "unknown", "launch_safety_state": "dead"},
        },
        {"tdue", "talive-launch", "trecent"},
        deps=TerminalDiagnosticsSnapshotDeps(
            state_lock=_lock,
            load_state=lambda: state,
            scan_completed_log_for_crash=lambda task: (False, ""),
            diagnose_terminal=lambda task: {
                "is_crash": False,
                "reason": "done",
                "success_marker": "DONE",
            },
            discover_result_artifacts=lambda *args, **kwargs: [],
            notify=lambda *args, **kwargs: None,
            now=lambda: 500.0,
            node_down_requeue_s=30,
        ),
    )

    assert set(out) == {"tdue"}
    assert out["tdue"]["diagnosis"]["success_marker"] == "DONE"


def test_batch_check_running_routes_only_probed_running_tasks():
    calls = []
    state = {
        "tasks": [
            {"id": "talive", "status": "running"},
            {"id": "tdead", "status": "running"},
            {"id": "tunknown", "status": "running"},
            {"id": "tskipped", "status": "running"},
            {"id": "tdone", "status": "done"},
        ]
    }

    batch_check_running(
        state,
        probe_results={
            "talive": {"state": "alive"},
            "tdead": {"state": "dead"},
            "tunknown": {"state": "unknown"},
        },
        probed_task_ids={"talive", "tdead", "tunknown"},
        terminal_diagnostics={"tdead": {"diagnosis": {}}},
        deps=BatchCheckRunningDeps(
            backend_batch_probe=lambda state: {},
            running_lifecycle_deps=lambda: "lifecycle",
            handle_unknown_probe_result=lambda task, state, res, diagnostics, **kwargs: calls.append(
                ("unknown", task["id"], diagnostics, kwargs["deps"])
            ),
            clear_probe_unknown=lambda task: calls.append(("clear", task["id"])) or {"sync": True},
            handle_dead_probe_result=lambda task, state, res, sync, diagnostics, **kwargs: calls.append(
                ("dead", task["id"], sync, diagnostics, kwargs["deps"])
            ),
            apply_alive_probe_result=lambda task, res, sync, **kwargs: calls.append(("alive", task["id"], sync, kwargs["deps"])),
        ),
    )

    assert calls == [
        ("clear", "talive"),
        ("alive", "talive", {"sync": True}, "lifecycle"),
        ("clear", "tdead"),
        ("dead", "tdead", {"sync": True}, {"tdead": {"diagnosis": {}}}, "lifecycle"),
        (
            "unknown",
            "tunknown",
            {"tdead": {"diagnosis": {}}},
            "lifecycle",
        ),
    ]


def test_update_running_tasks_chains_probe_and_eta_passes():
    calls = []
    state = {"tasks": []}

    update_running_tasks(
        state,
        probe_results={"t1": {"state": "alive"}},
        probed_task_ids={"t1"},
        eta_tail_outputs={"n1": "tail"},
        eta_probed_task_ids={"t1"},
        terminal_diagnostics={"t1": {}},
        deps=UpdateRunningTasksDeps(
            now=lambda: 100.0 + len(calls),
            running_probe_due=lambda state, now=None: True,
            batch_check_running=lambda *args, **kwargs: calls.append(("batch", kwargs)),
            eta_refresh_due=lambda state, now=None: True,
            refresh_eta_from_logs=lambda *args, **kwargs: calls.append(("eta", kwargs)),
        ),
    )

    assert calls[0] == (
        "batch",
        {
            "probe_results": {"t1": {"state": "alive"}},
            "probed_task_ids": {"t1"},
            "terminal_diagnostics": {"t1": {}},
        },
    )
    assert calls[1] == (
        "eta",
        {"tail_outputs": {"n1": "tail"}, "probed_task_ids": {"t1"}},
    )
    assert state["_last_running_probe_at"] == 101.0
    assert state["_last_eta_refresh_at"] == 102.0


def test_update_running_tasks_can_defer_eta_to_background_worker():
    calls = []
    state = {"tasks": []}

    update_running_tasks(
        state,
        defer_eta_refresh=True,
        deps=UpdateRunningTasksDeps(
            now=lambda: 100.0,
            running_probe_due=lambda state, now=None: False,
            batch_check_running=lambda *args, **kwargs: calls.append("probe"),
            eta_refresh_due=lambda state, now=None: True,
            refresh_eta_from_logs=lambda *args, **kwargs: calls.append("eta"),
        ),
    )

    assert calls == []
    assert "_last_eta_refresh_at" not in state


def test_update_running_tasks_commits_precomputed_probe_after_timestamp_race():
    calls = []
    timestamps = iter((100.0, 101.0))
    state = {
        "tasks": [{"id": "t1", "status": "running"}],
        "_last_running_probe_at": 99.0,
    }

    update_running_tasks(
        state,
        probe_results={"t1": {"state": "dead"}},
        probed_task_ids={"t1"},
        terminal_diagnostics={"t1": {"diagnosis": {"is_crash": False}}},
        deps=UpdateRunningTasksDeps(
            now=lambda: next(timestamps),
            running_probe_due=lambda *_args, **_kwargs: False,
            batch_check_running=lambda *args, **kwargs: calls.append((args, kwargs)),
            eta_refresh_due=lambda *_args, **_kwargs: False,
            refresh_eta_from_logs=lambda *args, **kwargs: None,
        ),
    )

    assert len(calls) == 1
    assert calls[0][1] == {
        "probe_results": {"t1": {"state": "dead"}},
        "probed_task_ids": {"t1"},
        "terminal_diagnostics": {"t1": {"diagnosis": {"is_crash": False}}},
    }
    assert state["_last_running_probe_at"] == 101.0


def test_update_running_tasks_does_not_probe_under_lock_after_empty_snapshot():
    calls = []
    state = {
        "tasks": [{"id": "t1", "status": "running"}],
        "_last_running_probe_at": 99.0,
    }

    update_running_tasks(
        state,
        probe_results={},
        probed_task_ids=set(),
        deps=UpdateRunningTasksDeps(
            now=lambda: 100.0,
            running_probe_due=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("outside-lock empty snapshot is authoritative")
            ),
            batch_check_running=lambda *args, **kwargs: calls.append(
                ("probe", args, kwargs)
            ),
            eta_refresh_due=lambda *_args, **_kwargs: False,
            refresh_eta_from_logs=lambda *args, **kwargs: calls.append(
                ("eta", args, kwargs)
            ),
        ),
    )

    assert calls == []
    assert state["_last_running_probe_at"] == 99.0


def test_update_running_tasks_terminal_retry_does_not_delay_full_probe():
    calls = []
    state = {
        "tasks": [{"id": "pending", "status": "running"}],
        "_last_running_probe_at": 50.0,
    }
    retry_probe = {
        "pending": {
            "state": "dead",
            "terminal_diagnosis_deferred": True,
            "terminal_retry_only": True,
        }
    }

    update_running_tasks(
        state,
        probe_results=retry_probe,
        probed_task_ids={"pending"},
        terminal_diagnostics={"pending": {"diagnosis": {"is_crash": False}}},
        deps=UpdateRunningTasksDeps(
            now=lambda: 100.0,
            running_probe_due=lambda *_args, **_kwargs: False,
            batch_check_running=lambda *args, **kwargs: calls.append(kwargs),
            eta_refresh_due=lambda *_args, **_kwargs: False,
            refresh_eta_from_logs=lambda *args, **kwargs: None,
        ),
    )

    assert calls == [
        {
            "probe_results": retry_probe,
            "probed_task_ids": {"pending"},
            "terminal_diagnostics": {
                "pending": {"diagnosis": {"is_crash": False}}
            },
        }
    ]
    assert state["_last_running_probe_at"] == 50.0


def test_eta_tail_outputs_by_node_uses_freqduet_expanded_tail():
    captured = {}
    by_node = {
        "node001": [
            ({"id": "t1", "project": "FreqDuet"}, "/tmp/freq.log"),
            ({"id": "t2", "project": "Other"}, "/tmp/other.log"),
        ]
    }

    def run_on(node, cmd, **kwargs):
        captured["node"] = node
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return 0, "ok", ""

    out = eta_tail_outputs_by_node(
        by_node,
        deps=EtaTailOutputDeps(
            node_is_windows=lambda node: False,
            ps_quote=lambda value: repr(value),
            windows_tail_ps=lambda *a, **k: "",
            windows_path_for_task=lambda task, path: path,
            run_windows_ps=lambda *a, **k: (1, "", ""),
            run_on=run_on,
            timeout_s=7,
        ),
    )

    assert out == {"node001": "ok"}
    assert captured["node"] == "node001"
    assert "grep -aE '^(Shard jobs|SKIP |DONE )' /tmp/freq.log" in captured["cmd"]
    assert "tail -c 65536 /tmp/freq.log" in captured["cmd"]
    assert "tail -c 4096 /tmp/other.log" in captured["cmd"]
    assert captured["kwargs"] == {"timeout": 7, "check": False}


def test_eta_tail_outputs_by_node_preserves_bapr_stage_markers():
    captured = {}
    by_node = {
        "node001": [
            ({"id": "t1", "project": "BAPR"}, "/tmp/bapr.log"),
        ]
    }

    def run_on(node, cmd, **kwargs):
        captured["cmd"] = cmd
        return 0, "ok", ""

    out = eta_tail_outputs_by_node(
        by_node,
        deps=EtaTailOutputDeps(
            node_is_windows=lambda node: False,
            ps_quote=lambda value: repr(value),
            windows_tail_ps=lambda *a, **k: "",
            windows_path_for_task=lambda task, path: path,
            run_windows_ps=lambda *a, **k: (1, "", ""),
            run_on=run_on,
            timeout_s=7,
        ),
    )

    assert out == {"node001": "ok"}
    assert "TRAIN SUBPROCESS:" in captured["cmd"]
    assert "Training complete! Total time:" in captured["cmd"]
    assert "tail -c 65536 /tmp/bapr.log" in captured["cmd"]


def test_effective_elapsed_s_preserves_slurm_pending_semantics():
    assert effective_elapsed_s({"started_at": 80.0}, now=lambda: 100.0) == 20.0
    assert effective_elapsed_s(
        {"started_at": 1.0, "slurm_job_id": 42},
        now=lambda: 100.0,
    ) == 0.0
    assert effective_elapsed_s(
        {"started_at": 1.0, "slurm_job_id": 42, "actual_started_at": 70.0},
        now=lambda: 100.0,
    ) == 30.0
