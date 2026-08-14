from __future__ import annotations

import subprocess

from skill.scheduler_data_motion.rsync import run_local_rsync, run_remote_rsync_command


class _RunResult:
    def __init__(self, returncode: int = 0, stderr: str = ""):
        self.returncode = returncode
        self.stderr = stderr


def test_run_local_rsync_preserves_custom_failure_and_timeout_labels():
    def fail(args, **kwargs):
        return _RunResult(returncode=23, stderr="disk full")

    ok, msg = run_local_rsync(
        ["rsync"],
        operation="rsync cwd",
        run_subprocess=fail,
        timeout_human=">10min",
        failure_word=" failed",
    )

    assert ok is False
    assert msg == "rsync cwd failed rc=23: disk full"

    def timeout(args, **kwargs):
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    ok, msg = run_local_rsync(
        ["rsync"],
        operation="rsync ckpt",
        rc_operation="rsync ckpt failed",
        timeout_operation="rsync ckpt",
        exception_operation="rsync ckpt",
        run_subprocess=timeout,
        timeout_s=600,
        timeout_human=">600s",
    )

    assert ok is False
    assert msg == "rsync ckpt timeout (>600s)"


def test_run_remote_rsync_command_formats_run_on_failure():
    def run_on(node, cmd, **kwargs):
        assert node == "relay"
        assert cmd == "rsync src dst"
        assert kwargs == {"timeout": 7, "check": False}
        return 12, "", "connection refused"

    ok, msg = run_remote_rsync_command(
        node="relay",
        command="rsync src dst",
        run_on=run_on,
        operation="rsync ckpt relay→target",
        timeout_s=7,
    )

    assert ok is False
    assert msg == "rsync ckpt relay→target rc=12: connection refused"
