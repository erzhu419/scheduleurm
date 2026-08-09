from __future__ import annotations

from algorithm.experiments import critical_gpu_loaded_wave_runner as runner


def test_busy_preflight_does_not_launch(monkeypatch):
    monkeypatch.setattr(
        runner,
        "_gpu_idle_snapshot",
        lambda _spec: {
            "ready": False,
            "selected_gpus": [{"index": 0, "used_mb": 2048, "util_pct": 100}],
        },
    )

    def forbidden(**_kwargs):
        raise AssertionError("measurement campaign must not launch")

    monkeypatch.setattr(runner, "build_loaded_completion_campaign", forbidden)

    report = runner.run_safe_loaded_wave(node="node007", wave=5, allow_launch=True)

    assert report["status"] == "WAIT_GPU_IDLE"
    assert report["pass"] is False
    assert report["launched"] is False
    assert report["ordinary_running_tasks_touched"] is False


def test_idle_preflight_launches_exact_requested_wave(monkeypatch):
    monkeypatch.setattr(
        runner,
        "_gpu_idle_snapshot",
        lambda _spec: {"ready": True, "selected_gpus": []},
    )
    calls = []

    def campaign(**kwargs):
        calls.append(kwargs)
        return {"status": "PASS", "pass": True, "rows": [{"scenario_id": "a"}]}

    monkeypatch.setattr(runner, "build_loaded_completion_campaign", campaign)
    monkeypatch.setattr(
        runner,
        "_prelaunch_assigned_gpu_audit",
        lambda **_kwargs: {
            "audit_ready": True,
            "assigned_gpu_idle": True,
            "all_registered_gpus_idle": True,
        },
    )
    monkeypatch.setattr(
        runner,
        "_continuous_other_gpu_audit",
        lambda **_kwargs: {
            "audit_ready": True,
            "target_interval_covered": True,
            "other_registered_gpus_idle": True,
            "controlled_process_groups_observed": True,
            "no_unapproved_compute_processes": True,
        },
    )

    report = runner.run_safe_loaded_wave(
        node="node007",
        wave=13,
        allow_launch=True,
        keep_remote_output=True,
    )

    assert report["status"] == "PASS"
    assert report["pass"] is True
    assert report["launched"] is True
    assert report["row_prelaunch_audit"]["pass"] is True
    assert calls == [
        {
            "node": "node007",
            "wave": 13,
            "allow_launch": True,
            "keep_remote_output": True,
        }
    ]


def test_cross_gpu_prelaunch_contamination_stops_next_wave(monkeypatch):
    monkeypatch.setattr(
        runner,
        "_gpu_idle_snapshot",
        lambda _spec: {"ready": True, "selected_gpus": []},
    )
    monkeypatch.setattr(
        runner,
        "build_loaded_completion_campaign",
        lambda **_kwargs: {
            "status": "PASS",
            "pass": True,
            "rows": [{"scenario_id": "a"}],
        },
    )
    monkeypatch.setattr(
        runner,
        "_prelaunch_assigned_gpu_audit",
        lambda **_kwargs: {
            "audit_ready": True,
            "assigned_gpu_idle": True,
            "all_registered_gpus_idle": False,
            "nonidle_registered_gpus": [{"gpu": 1}],
        },
    )
    monkeypatch.setattr(
        runner,
        "_continuous_other_gpu_audit",
        lambda **_kwargs: {
            "audit_ready": True,
            "target_interval_covered": True,
            "other_registered_gpus_idle": True,
            "controlled_process_groups_observed": True,
            "no_unapproved_compute_processes": True,
        },
    )

    report = runner.run_safe_loaded_wave(node="node007", wave=5, allow_launch=True)

    assert report["status"] == "ROW_PREFLIGHT_AUDIT_FAILED"
    assert report["pass"] is False
    assert report["launched"] is True
    assert report["row_prelaunch_audit"]["pass"] is False


def test_manifest_mode_never_runs_idle_probe(monkeypatch):
    monkeypatch.setattr(
        runner,
        "_gpu_idle_snapshot",
        lambda _spec: (_ for _ in ()).throw(AssertionError("idle probe should not run")),
    )
    monkeypatch.setattr(
        runner,
        "build_loaded_completion_campaign",
        lambda **_kwargs: {"status": "MANIFEST_ONLY", "pass": False},
    )

    report = runner.run_safe_loaded_wave(node="node007", wave=6, allow_launch=False)

    assert report["status"] == "MANIFEST_ONLY"
    assert report["launched"] is False
