from __future__ import annotations

from types import SimpleNamespace
import subprocess

from algorithm.experiments import remote_workload_selected_profile_probe as probe


def test_direct_ssh_capture_retries_transient_kex_failure(tmp_path, monkeypatch):
    responses = [
        SimpleNamespace(
            returncode=255,
            stdout="",
            stderr="kex_exchange_identification: Connection closed by remote host",
        ),
        SimpleNamespace(returncode=0, stdout="ready", stderr=""),
    ]
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return responses.pop(0)

    monkeypatch.setattr(probe, "_scheduler_node", lambda node: False)
    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    monkeypatch.setattr(probe.time, "sleep", lambda seconds: None)

    rc, out, err = probe._run_remote_capture(
        "jtl311linux",
        "true",
        tmp_path / "capture",
        timeout_s=30,
    )

    assert rc == 0
    assert out == "ready"
    assert err == ""
    assert len(calls) == 2


def test_direct_ssh_capture_does_not_retry_workload_failure(tmp_path, monkeypatch):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(
            returncode=7,
            stdout="",
            stderr="workload assertion failed",
        )

    monkeypatch.setattr(probe, "_scheduler_node", lambda node: False)
    monkeypatch.setattr(probe.subprocess, "run", fake_run)

    rc, _, _ = probe._run_remote_capture(
        "jtl311linux",
        "false",
        tmp_path / "capture",
        timeout_s=30,
    )

    assert rc == 7
    assert len(calls) == 1


def test_scheduler_capture_survives_four_consecutive_banner_timeouts(monkeypatch):
    calls = []

    class FakeScheduler:
        @staticmethod
        def run_on(node, rendered, timeout, check):
            calls.append((node, rendered, timeout, check))
            if len(calls) <= 4:
                return 255, "", "Connection timed out during banner exchange"
            return 0, "ready", ""

    monkeypatch.setattr(probe.time, "sleep", lambda seconds: None)
    rc, out, err = probe._scheduler_run_on_retry(
        FakeScheduler(), "jtl110gpu", "true", timeout_s=30
    )

    assert (rc, out, err) == (0, "ready", "")
    assert len(calls) == 5


def test_detached_remote_process_survives_poll_transport_boundary(
    tmp_path,
    monkeypatch,
):
    commands = []
    status_calls = 0

    def fake_capture(node, command, prefix, *, timeout_s):
        nonlocal status_calls
        commands.append(command)
        if "nohup setsid" in command:
            return 0, "", ""
        if "__SCHEDULEURM_DETACHED_RUNNING__" in command:
            status_calls += 1
            if status_calls == 1:
                return 3, "__SCHEDULEURM_DETACHED_RUNNING__\n", ""
            return 0, "__SCHEDULEURM_DETACHED_RC__ 0\n", ""
        if command.startswith("cat "):
            return 0, "ScheduleurmCompletionModel {}\n", ""
        raise AssertionError(command)

    monkeypatch.setattr(probe, "_run_remote_capture", fake_capture)
    monkeypatch.setattr(probe.time, "sleep", lambda seconds: None)

    proc = probe._remote_detached_popen(
        "node001",
        "python3 workload.py",
        run_name="unit-detached",
        local_prefix=tmp_path / "detached",
        timeout_s=60,
    )
    output, stderr = proc.communicate(timeout=30)

    assert stderr is None
    assert proc.returncode == 0
    assert "ScheduleurmCompletionModel" in output
    assert status_calls == 2
    assert "kill -0" in commands[0]


def test_detached_returncode_parser_fails_closed_without_marker():
    assert probe._detached_returncode("noise") is None
    assert probe._detached_returncode(
        "__SCHEDULEURM_DETACHED_RC__ 139"
    ) == 139


def test_coordinated_profile_uses_transport_safe_detached_process(tmp_path, monkeypatch):
    calls = []

    class FakeProcess:
        returncode = 0

        def communicate(self, timeout):
            return (
                "__SCHEDULEURM_LOG_BEGIN__ gpu0_0\n"
                "Step 1/2 rate=1.0 step/s\n"
                "__SCHEDULEURM_LOG_END__ gpu0_0\n"
                "__SCHEDULEURM_RC__ gpu0_0 0\n",
                None,
            )

    monkeypatch.setattr(probe, "_write_remote_text", lambda *args, **kwargs: None)
    monkeypatch.setattr(probe, "_scheduler_node", lambda node: False)

    def fake_detached(node, shell_cmd, *, run_name, local_prefix, timeout_s):
        calls.append((node, shell_cmd, run_name, local_prefix, timeout_s))
        return FakeProcess()

    monkeypatch.setattr(probe, "_remote_detached_popen", fake_detached)
    rows = probe._run_coordinated_profile(
        node="jtl110gpu",
        run_id="unit-run",
        phase="profile_1_per_gpu",
        launch_rows=[
            (
                0,
                0,
                {"values": {"index": 0}, "seed": 7, "run_name": "unit-task"},
                "python workload.py",
            )
        ],
        raw_dir=tmp_path,
        timeout_s=60,
        unit="step",
    )

    assert len(calls) == 1
    assert calls[0][2].startswith("coordinated_unit-run_profile_1_per_gpu")
    assert rows[0]["returncode"] == 0
    assert rows[0]["rate"] == 1.0


def test_coordinated_script_staggers_initialization_and_barriers_builtin_tasks(tmp_path):
    script = probe._coordinated_remote_script(
        "/tmp/unit",
        [
            (0, 0, {"values": {"coordinated_start_barrier": "1", "profile": 3, "admission_stagger_s": 0.75}}, "python a.py"),
            (0, 1, {"values": {"coordinated_start_barrier": "1", "profile": 3, "admission_stagger_s": 0.75}}, "python b.py"),
        ],
    )

    assert "export SCHEDULEURM_ADMISSION_DELAY_S=0.75" in script
    assert "export SCHEDULEURM_READY_FILE" in script
    assert "export OMP_NUM_THREADS=1" in script
    assert "__SCHEDULEURM_BARRIER__ ready=" in script
    path = tmp_path / "coordinated.sh"
    path.write_text(script, encoding="utf-8")
    subprocess.run(["bash", "-n", str(path)], check=True)


def test_coordinated_script_does_not_barrier_external_rl_workload():
    script = probe._coordinated_remote_script(
        "/tmp/unit",
        [(0, 0, {"values": {"coordinated_start_barrier": "0"}}, "python rl.py")],
    )

    assert "SCHEDULEURM_READY_FILE" not in script
    assert "__SCHEDULEURM_BARRIER__" not in script
    assert "export OMP_NUM_THREADS=1" in script


def test_global_staged_admission_delay_is_exported_for_high_profile_rl():
    script = probe._coordinated_remote_script(
        "/tmp/unit",
        [(
            3,
            4,
            {
                "values": {
                    "coordinated_start_barrier": "0",
                    "profile": 5,
                    "index": 19,
                    "admission_stagger_s": 30.0,
                    "admission_stagger_axis": "global_index",
                    "admission_stagger_min_profile": 5,
                }
            },
            "python rl.py",
        )],
    )

    assert "export SCHEDULEURM_ADMISSION_DELAY_S=570" in script
    assert "__SCHEDULEURM_BARRIER__" not in script
