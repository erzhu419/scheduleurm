def test_update_running_tasks_uses_precomputed_terminal_diagnosis(monkeypatch, sch):
    task = {
        "id": "tdead",
        "status": "running",
        "node": "node006",
        "log_path": "/tmp/sched_tdead.log",
        "started_at": 100.0,
        "signature": "proj/run",
        "project": "proj",
        "cmd": "python train.py",
        "remote_pids": [123],
        "peak_vram_mb": 0,
        "peak_ram_mb": 0,
        "cpu_cores": 1,
    }
    state = {"tasks": [task], "next_id": 2}
    probe_results = {
        "tdead": {
            "state": "dead",
            "alive_pids": [],
            "vram_mb": 0,
            "ram_mb": 0,
            "pcpu": 0.0,
        }
    }
    precomputed = {
        "fingerprint": sch._terminal_diagnostic_fingerprint(task),
        "diagnosis": {
            "is_crash": False,
            "reason": "precomputed normal exit",
            "tail": "DONE",
            "lifetime_s": 123,
            "log_size": 4,
            "log_path": task["log_path"],
            "success_marker": "DONE",
        },
        "result_artifacts": [],
    }

    def should_not_run(_task):
        raise AssertionError("terminal diagnosis should have been precomputed outside the lock")

    def should_not_index_results(_task):
        raise AssertionError("result artifacts should have been precomputed outside the lock")

    monkeypatch.setattr(sch, "_diagnose_terminal", should_not_run, raising=False)
    monkeypatch.setattr(sch, "_record_result_artifacts", should_not_index_results, raising=False)
    monkeypatch.setattr(sch, "_release_task_claims_and_intents", lambda *a, **k: 0, raising=False)
    monkeypatch.setattr(sch, "history_record", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(sch, "runtime_history_record", lambda *a, **k: None, raising=False)

    sch.update_running_tasks(
        state,
        probe_results=probe_results,
        probed_task_ids={"tdead"},
        terminal_diagnostics={"tdead": precomputed},
    )

    assert task["status"] == "done"
    assert task["_diagnosis"]["reason"] == "precomputed normal exit"


def test_precomputed_terminal_diagnosis_fingerprint_mismatch_defers(monkeypatch, sch):
    task = {
        "id": "tdead",
        "status": "running",
        "node": "node006",
        "log_path": "/tmp/sched_tdead.log",
        "started_at": 100.0,
        "signature": "proj/run",
        "project": "proj",
        "cmd": "python train.py",
        "remote_pids": [123],
        "peak_vram_mb": 0,
        "peak_ram_mb": 0,
        "cpu_cores": 1,
    }
    state = {"tasks": [task], "next_id": 2}
    probe_results = {
        "tdead": {
            "state": "dead",
            "alive_pids": [],
            "vram_mb": 0,
            "ram_mb": 0,
            "pcpu": 0.0,
        }
    }
    calls = {"diagnose": 0}

    def fallback_diag(_task):
        calls["diagnose"] += 1
        return {
            "is_crash": False,
            "reason": "fallback normal exit",
            "tail": "DONE",
            "lifetime_s": 123,
            "log_size": 4,
            "log_path": task["log_path"],
            "success_marker": "DONE",
        }

    monkeypatch.setattr(sch, "_diagnose_terminal", fallback_diag, raising=False)
    monkeypatch.setattr(sch, "_record_result_artifacts", lambda _task: [], raising=False)
    monkeypatch.setattr(sch, "_release_task_claims_and_intents", lambda *a, **k: 0, raising=False)
    monkeypatch.setattr(sch, "history_record", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(sch, "runtime_history_record", lambda *a, **k: None, raising=False)

    sch.update_running_tasks(
        state,
        probe_results=probe_results,
        probed_task_ids={"tdead"},
        terminal_diagnostics={"tdead": {"fingerprint": ("wrong",)}},
    )

    assert calls["diagnose"] == 0
    assert task["status"] == "running"
    assert task["terminal_transition_deferred"] is True
    assert "outside state lock" in task["last_block_reason"]
