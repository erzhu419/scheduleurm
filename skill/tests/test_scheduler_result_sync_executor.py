from __future__ import annotations

import base64
import subprocess

from skill.scheduler_result_sync_executor import (
    ResultSyncExecutorDeps,
    WindowsResultSyncDeps,
    sync_one_result,
    sync_windows_result,
)


class _Completed:
    def __init__(self, returncode=0, stderr=""):
        self.returncode = returncode
        self.stderr = stderr


class _Pipe:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _FakePopen:
    def __init__(self, args, stdout=None, stderr=None, returncode=0, ssh_err=b""):
        self.args = list(args)
        self.stdout = _Pipe()
        self.returncode = returncode
        self._ssh_err = ssh_err
        self.killed = False

    def communicate(self, timeout=None):
        return b"", self._ssh_err

    def kill(self):
        self.killed = True


def _deps(
    *,
    calls=None,
    relay_node=None,
    subprocess_run=None,
    node_is_windows=None,
    sync_windows_result=None,
    run_on=None,
    timeout_s=1800,
):
    calls = calls if calls is not None else []

    def default_run_on(node, cmd, **kwargs):
        calls.append(("run_on", node, cmd, kwargs))
        return 0, "", ""

    def default_subprocess_run(args, **kwargs):
        calls.append(("subprocess_run", list(args), kwargs))
        return _Completed(0)

    def default_sync_windows(candidate):
        calls.append(("sync_windows_result", dict(candidate)))
        return True, "windows"

    return ResultSyncExecutorDeps(
        node_is_windows=node_is_windows or (lambda node: False),
        sync_windows_result=sync_windows_result or default_sync_windows,
        remote_path_for_node=lambda node, path: f"/remote/{node}{path}",
        relay_node_for_node=lambda node: relay_node,
        relay_path_for_node=lambda node, path: f"/relay/{node}{path}",
        relay_ssh_target_for_node=lambda node: f"user@{node}",
        run_on=run_on or default_run_on,
        ssh_rsync_shell_for_node=lambda node: f"ssh-shell-{node}",
        rsync_path_for_node=lambda node, path: f"user@{node}:{path}",
        ssh_target_for_node=lambda node: f"user@{node}",
        result_sync_timeout_s=timeout_s,
        subprocess_run=subprocess_run or default_subprocess_run,
    )


def _windows_deps(*, calls=None, popen=None, subprocess_run=None, mkdir=None, timeout_s=1800):
    calls = calls if calls is not None else []

    def default_mkdir(path):
        calls.append(("mkdir", path))

    def default_popen(args, **kwargs):
        calls.append(("popen", list(args), kwargs))
        return _FakePopen(args)

    def default_run(args, **kwargs):
        calls.append(("subprocess_run", list(args), kwargs))
        return _Completed(0)

    return WindowsResultSyncDeps(
        windows_path_for_node=lambda node, path: f"C:\\results{path}",
        ps_quote=lambda text: f"'{text}'",
        ssh_base_args=lambda node: ["ssh", node],
        result_sync_timeout_s=timeout_s,
        make_local_dir=mkdir or default_mkdir,
        popen=popen or default_popen,
        subprocess_run=subprocess_run or default_run,
        pipe="PIPE",
    )


def test_direct_rsync_uses_remote_mapping_and_creates_local_dir(tmp_path):
    calls = []
    dst = tmp_path / "result"

    ok, msg = sync_one_result(
        {"node": "node1", "result_dir": "/runs/a", "local_result_dir": str(dst)},
        deps=_deps(calls=calls),
    )

    assert (ok, msg) == (True, "ok")
    assert dst.is_dir()
    assert calls == [
        (
            "subprocess_run",
            [
                "rsync", "-az", "--partial", "-e", "ssh-shell-node1",
                "user@node1:/remote/node1/runs/a/",
                str(dst) + "/",
            ],
            {"capture_output": True, "text": True, "timeout": 1800},
        )
    ]


def test_windows_node_delegates_to_windows_sync(tmp_path):
    calls = []
    candidate = {"node": "win1", "result_dir": "/runs/a", "local_result_dir": str(tmp_path / "a")}

    ok, msg = sync_one_result(
        candidate,
        deps=_deps(calls=calls, node_is_windows=lambda node: node == "win1"),
    )

    assert (ok, msg) == (True, "windows")
    assert calls == [("sync_windows_result", candidate)]


def test_relay_sync_pulls_to_relay_then_to_local(tmp_path):
    calls = []
    dst = tmp_path / "result"

    ok, msg = sync_one_result(
        {"node": "node1", "result_dir": "/runs/a", "local_result_dir": str(dst)},
        deps=_deps(calls=calls, relay_node="relay1"),
    )

    assert (ok, msg) == (True, "ok")
    assert calls[0] == (
        "run_on",
        "relay1",
        "mkdir -p /relay/node1/remote/node1/runs/a",
        {"timeout": 10, "check": False},
    )
    assert calls[1][0:2] == ("run_on", "relay1")
    assert "user@node1:/remote/node1/runs/a/" in calls[1][2]
    assert calls[2] == (
        "subprocess_run",
        [
            "rsync", "-az", "--partial", "-e", "ssh-shell-relay1",
            "user@relay1:/relay/node1/remote/node1/runs/a/",
            str(dst) + "/",
        ],
        {"capture_output": True, "text": True, "timeout": 1800},
    )


def test_multi_result_dirs_use_explicit_local_dirs(tmp_path):
    calls = []
    dst_a = tmp_path / "a"
    dst_b = tmp_path / "b"

    ok, msg = sync_one_result(
        {
            "node": "node1",
            "result_dirs": ["/runs/a", "/runs/b"],
            "local_result_dirs": [str(dst_a), str(dst_b)],
            "result_dir": "/runs/a",
            "local_result_dir": str(tmp_path / "unused"),
        },
        deps=_deps(calls=calls),
    )

    assert (ok, msg) == (True, "synced 2 dirs")
    subprocess_calls = [call for call in calls if call[0] == "subprocess_run"]
    assert subprocess_calls[0][1][-1] == str(dst_a) + "/"
    assert subprocess_calls[1][1][-1] == str(dst_b) + "/"


def test_multi_result_failure_reports_partial_success(tmp_path):
    calls = []
    outcomes = [_Completed(0), _Completed(23, "broken")]

    def fake_run(args, **kwargs):
        calls.append(("subprocess_run", list(args), kwargs))
        return outcomes.pop(0)

    ok, msg = sync_one_result(
        {
            "node": "node1",
            "result_dirs": ["/runs/a", "/runs/b"],
            "local_result_dir_base": str(tmp_path / "base"),
            "result_dir": "/runs/a",
            "local_result_dir": str(tmp_path / "unused"),
        },
        deps=_deps(calls=calls, subprocess_run=fake_run),
    )

    assert ok is False
    assert "synced 1/2 dirs" in msg
    assert "/runs/b: rsync rc=23: broken" in msg


def test_direct_rsync_timeout_reports_configured_timeout(tmp_path):
    def fake_run(args, **kwargs):
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    ok, msg = sync_one_result(
        {"node": "node1", "result_dir": "/runs/a", "local_result_dir": str(tmp_path / "a")},
        deps=_deps(subprocess_run=fake_run, timeout_s=7),
    )

    assert ok is False
    assert msg == "rsync timeout (>7s)"


def test_sync_windows_result_streams_remote_tar_to_local_extract():
    calls = []

    ok, msg = sync_windows_result(
        {"node": "win1", "result_dir": "/runs/a", "local_result_dir": "/tmp/out"},
        deps=_windows_deps(calls=calls),
    )

    assert (ok, msg) == (True, "ok")
    assert calls[0] == ("mkdir", "/tmp/out/")
    popen_args = calls[1][1]
    assert popen_args[:2] == ["ssh", "win1"]
    encoded = popen_args[-1]
    script = base64.b64decode(encoded).decode("utf-16le")
    assert "$src='C:\\results/runs/a'" in script
    assert "tar -cf - -C $src ." in script
    assert calls[2][1] == ["tar", "-xf", "-", "-C", "/tmp/out/"]
    assert calls[2][2]["stdin"].closed is True
    assert calls[2][2]["timeout"] == 1800


def test_sync_windows_result_reports_mkdir_failure():
    def fail_mkdir(path):
        raise OSError("readonly")

    ok, msg = sync_windows_result(
        {"node": "win1", "result_dir": "/runs/a", "local_result_dir": "/tmp/out"},
        deps=_windows_deps(mkdir=fail_mkdir),
    )

    assert ok is False
    assert msg == "mkdir local target failed: readonly"


def test_sync_windows_result_reports_ssh_tar_failure():
    def fake_popen(args, **kwargs):
        return _FakePopen(args, returncode=44, ssh_err=b"missing result_dir")

    ok, msg = sync_windows_result(
        {"node": "win1", "result_dir": "/runs/missing", "local_result_dir": "/tmp/out"},
        deps=_windows_deps(popen=fake_popen),
    )

    assert ok is False
    assert msg == "Windows result tar ssh rc=44: missing result_dir"


def test_sync_windows_result_reports_local_tar_failure():
    def fake_run(args, **kwargs):
        return _Completed(2, "tar broke")

    ok, msg = sync_windows_result(
        {"node": "win1", "result_dir": "/runs/a", "local_result_dir": "/tmp/out"},
        deps=_windows_deps(subprocess_run=fake_run),
    )

    assert ok is False
    assert msg == "local result tar extract rc=2: tar broke"
