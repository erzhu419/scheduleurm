from __future__ import annotations

from pathlib import Path

import pytest

from algorithm.experiments import durable_remote_command as durable


def test_durable_remote_command_launches_once_and_fetches_result(tmp_path, monkeypatch):
    polls = iter(("RUNNING", "DONE"))
    calls = []

    def fake_capture(node, command, prefix, *, timeout_s):
        calls.append(command)
        if "__DURABLE_STATE__ LAUNCHED" in command:
            return 0, "__DURABLE_STATE__ LAUNCHED\n__DURABLE_PID__ 123\n", ""
        if command.startswith("if [ ! -d"):
            state = next(polls)
            if state == "DONE":
                return 0, "__DURABLE_STATE__ DONE\n__DURABLE_RC__ 0\n__DURABLE_PID__ 123\n", ""
            return 0, "__DURABLE_STATE__ RUNNING\n__DURABLE_PID__ 123\n", ""
        if "cat --" in command and command.endswith("/stdout"):
            return 0, "payload\n", ""
        if "cat --" in command and command.endswith("/stderr"):
            return 0, "warning\n", ""
        if command.startswith("rm -rf --"):
            return 0, "", ""
        raise AssertionError(command)

    monkeypatch.setattr(durable, "_run_remote_capture", fake_capture)
    monkeypatch.setattr(durable.time, "sleep", lambda _seconds: None)

    rc, stdout, stderr, audit = durable.run_durable_remote_capture(
        "node",
        "sleep 1; echo payload",
        tmp_path / "run",
        run_id="registered-run-1",
        timeout_s=60,
        poll_interval_s=0,
    )

    assert (rc, stdout, stderr) == (0, "payload\n", "warning\n")
    assert audit["state"] == "DONE"
    assert audit["launch_attempt_count"] == 1
    assert audit["poll_count"] == 2
    assert audit["long_command_replayed"] is False
    assert audit["control_cleanup_ready"] is True
    assert sum("__DURABLE_STATE__ LAUNCHED" in command for command in calls) == 1


def test_durable_remote_command_recovers_ambiguous_transient_launch(tmp_path, monkeypatch):
    launch_count = 0
    poll_count = 0

    def fake_capture(node, command, prefix, *, timeout_s):
        nonlocal launch_count, poll_count
        if "__DURABLE_STATE__ LAUNCHED" in command:
            launch_count += 1
            if launch_count == 1:
                return 255, "", "Timeout, server host not responding."
            return 0, "__DURABLE_STATE__ LAUNCHED\n__DURABLE_PID__ 321\n", ""
        if command.startswith("if [ ! -d"):
            poll_count += 1
            if poll_count == 1:
                return 0, "__DURABLE_STATE__ ABSENT\n", ""
            return 0, "__DURABLE_STATE__ DONE\n__DURABLE_RC__ 0\n__DURABLE_PID__ 321\n", ""
        if "cat --" in command:
            return 0, "", ""
        if command.startswith("rm -rf --"):
            return 0, "", ""
        raise AssertionError(command)

    monkeypatch.setattr(durable, "_run_remote_capture", fake_capture)
    monkeypatch.setattr(durable.time, "sleep", lambda _seconds: None)

    rc, _, _, audit = durable.run_durable_remote_capture(
        "node",
        "echo once",
        tmp_path / "run",
        run_id="registered-run-2",
        timeout_s=60,
        poll_interval_s=0,
    )

    assert rc == 0
    assert audit["launch_attempt_count"] == 2
    assert audit["transient_transport_error_count"] == 1
    assert audit["long_command_replayed"] is False


def test_durable_remote_command_rejects_unsafe_run_id(tmp_path):
    with pytest.raises(ValueError, match="unsafe durable run id"):
        durable.run_durable_remote_capture(
            "node",
            "true",
            tmp_path / "run",
            run_id="../../unsafe",
            timeout_s=10,
        )


def test_launch_command_is_idempotent_detached_and_payload_encoded():
    rendered = durable._launch_command(
        control_dir="/tmp/scheduleurm_durable_control/run-1",
        shell_cmd="echo secret command",
    )

    assert "mkdir /tmp/scheduleurm_durable_control/run-1" in rendered
    assert "setsid bash" in rendered
    assert "echo secret command" not in rendered
    assert "__DURABLE_STATE__ RESUMED" in rendered
