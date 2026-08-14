from __future__ import annotations


def test_doctor_fix_adds_checkpoint_wait_for_file(tmp_path, sch):
    state = {
        "tasks": [{
            "id": "tEval",
            "status": "queued",
            "project": "simple-sac",
            "cmd": "python eval.py --ckpt-path ckpts/model.pt",
            "cwd": str(tmp_path),
        }]
    }

    issues, changed = sch._doctor_scan_state(state, fix=True)

    assert changed == 1
    assert issues[0]["code"] == "eval_missing_wait_for_file"
    assert issues[0]["severity"] == "fixed"
    assert state["tasks"][0]["wait_for_files"] == [str(tmp_path / "ckpts" / "model.pt")]
    assert "waiting for prerequisite" in state["tasks"][0]["last_block_reason"]


def test_doctor_fix_forces_simple_sac_large_data_local(monkeypatch, sch):
    released = []
    monkeypatch.setattr(
        sch,
        "_release_task_claims_and_intents",
        lambda task, **kwargs: released.append((task["id"], kwargs)) or 1,
        raising=False,
    )
    task = {
        "id": "tTrain",
        "status": "queued",
        "project": "simple-sac",
        "cmd": "python h2o+_bus_main.py --dataset-dir /data/bus_h2o --seed 1",
        "cwd": "/tmp/SimpleSAC",
        "node": "node001",
        "preferred_node": "node001",
        "gpu_idx": 0,
        "remote_pids": [123],
    }
    state = {"tasks": [task]}

    issues, changed = sch._doctor_scan_state(state, fix=True)

    assert changed == 1
    assert issues[0]["code"] == "simple_sac_large_data_not_local"
    assert task["require_node"] == "local"
    assert task["node"] is None
    assert task["gpu_idx"] is None
    assert task["remote_pids"] == []
    assert released and released[0][1]["extra_nodes"] == ["node001"]


def test_doctor_fix_clears_stale_queued_live_eta(monkeypatch, sch):
    seeded = []

    def fake_seed(state):
        seeded.append([task["id"] for task in state["tasks"]])
        state["tasks"][0]["eta_source"] = "runtime_history"

    monkeypatch.setattr(sch, "_seed_pending_eta_from_history", fake_seed, raising=False)
    task = {
        "id": "tEta",
        "status": "queued",
        "project": "eta",
        "eta_source": "tqdm",
        "runtime_est_source": "tqdm",
        "last_progress_line": "42%|####",
        "runtime_current_unit": 42,
    }
    state = {"tasks": [task]}

    issues, changed = sch._doctor_scan_state(state, fix=True)

    assert changed == 1
    assert issues[0]["code"] == "queued_stale_live_eta"
    assert "last_progress_line" not in task
    assert "runtime_current_unit" not in task
    assert task["eta_source"] == "runtime_history"
    assert seeded == [["tEta"]]
