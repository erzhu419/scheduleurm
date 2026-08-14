from __future__ import annotations

import subprocess
from pathlib import Path

from skill import scheduler_resume_ckpt_staging as resume_stage
from skill.scheduler_resume_ckpt_staging import ResumeCkptStagingDeps


class _RunResult:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _task(**overrides):
    task = {
        "id": "t1",
        "ckpt_dir": "/ckpt",
        "needs_resume": True,
    }
    task.update(overrides)
    return task


def _deps(
    *,
    nodes: dict | None = None,
    requires_resume: bool = True,
    stage_check_result: tuple[str, dict | None, str] | None = None,
    cache_hits: set[tuple] | None = None,
    success_marks: list[tuple] | None = None,
    cap_marks: list[tuple] | None = None,
    run_on_calls: list[tuple] | None = None,
    run_on_impl=None,
    subprocess_calls: list[list[str]] | None = None,
    subprocess_impl=None,
    windows_stage_calls: list[tuple] | None = None,
    windows_stage_result: tuple[bool, str] = (True, "windows copied"),
    windows_result: tuple[int, str, str] = (0, "OK", ""),
    relay_node: str | None = None,
    pair_shell: bool = True,
    relay_spool_root: str | None = None,
):
    nodes = nodes or {
        "local": {"host": None},
        "remote": {"host": "user@remote"},
        "hpc": {"host": "user@hpc"},
        "gpu2": {"host": "user@gpu2"},
        "win": {"host": "winhost", "windows": True},
        "src_remote": {"host": "user@src"},
        "win_src": {"host": "win-src", "windows": True},
    }
    cache_hits = cache_hits or set()
    success_marks = success_marks if success_marks is not None else []
    cap_marks = cap_marks if cap_marks is not None else []
    run_on_calls = run_on_calls if run_on_calls is not None else []
    subprocess_calls = subprocess_calls if subprocess_calls is not None else []
    windows_stage_calls = windows_stage_calls if windows_stage_calls is not None else []

    def remote_path(node, path):
        return path if node == "local" else f"/mnt/{node}{path}"

    def run_on(node, cmd, **kwargs):
        run_on_calls.append((node, cmd, kwargs))
        if run_on_impl:
            return run_on_impl(node, cmd, **kwargs)
        if cmd.startswith("test -d"):
            return 0, "", ""
        if "du -sm" in cmd:
            return 0, "42\n", ""
        if cmd.startswith("test -e"):
            return 0, "", ""
        if "ls -1" in cmd:
            return 0, "checkpoint.pt\n", ""
        return 0, "", ""

    def run_subprocess(args, **kwargs):
        subprocess_calls.append(list(args))
        if subprocess_impl:
            return subprocess_impl(args, **kwargs)
        return _RunResult()

    def stage_windows(src, node, dst, **kwargs):
        windows_stage_calls.append((src, node, dst, kwargs))
        return windows_stage_result

    def windows_path(node, path):
        return "C:\\stage\\" + node + "\\" + path.strip("/").replace("/", "\\")

    kwargs = {}
    if relay_spool_root is not None:
        kwargs["relay_spool_root"] = relay_spool_root
    return ResumeCkptStagingDeps(
        node_configs=nodes,
        task_requires_resume_scan=lambda task: requires_resume,
        resume_checkpoint_stage_check=lambda *args: (
            stage_check_result
            if stage_check_result is not None
            else ("needs_stage", {"node": "local", "path": "/ckpt/checkpoint.pt"}, "needs stage")
        ),
        resume_ckpt_stage_key=lambda source, target, ckpt, source_location=None: (
            source, target, ckpt),
        staging_cache_hit=lambda key: key in cache_hits,
        mark_stage_success=lambda key: success_marks.append(key),
        mark_cap_exceeded=lambda key: cap_marks.append(key),
        node_is_windows=lambda node: bool(nodes.get(node, {}).get("windows")),
        remote_path_for_node=remote_path,
        run_on=run_on,
        stage_local_dir_to_windows=stage_windows,
        windows_path_for_node=windows_path,
        run_windows_ps=lambda node, script, **kwargs: windows_result,
        ps_quote=lambda text: f"'{text}'",
        relay_node_for_node=lambda node: relay_node if node == "hpc" else None,
        relay_path_for_node=lambda node, path: f"/relay/{node}{path}",
        rsync_path_for_node=lambda node, path: path if node == "local" else f"{node}:{path}",
        ssh_rsync_shell_for_node=lambda node: f"ssh-shell-{node}",
        relay_ssh_target_for_node=lambda node: f"{node}-via-relay",
        rsync_shell_for_pair=lambda source, target: (
            f"pair-shell-{source}-{target}" if pair_shell else None
        ),
        run_subprocess=run_subprocess,
        **kwargs,
    )


def test_non_resume_task_short_circuits():
    def fail_run_on(*args, **kwargs):
        raise AssertionError("run_on should not be called")

    ok, msg = resume_stage.stage_resume_ckpt_for_launch(
        _task(),
        "remote",
        source_loc={"node": "local"},
        max_ckpt_mb=100,
        deps=_deps(requires_resume=False, run_on_impl=fail_run_on),
    )

    assert ok is True
    assert msg == "not a resume-scanned task"


def test_stage_check_ready_and_missing_source_short_circuit():
    ok, msg = resume_stage.stage_resume_ckpt_for_launch(
        _task(),
        "remote",
        max_ckpt_mb=100,
        deps=_deps(stage_check_result=("ready", None, "checkpoint already local")),
    )
    assert ok is True
    assert msg == "checkpoint already local"

    ok, msg = resume_stage.stage_resume_ckpt_for_launch(
        _task(),
        "remote",
        max_ckpt_mb=100,
        deps=_deps(stage_check_result=("missing", None, "no checkpoint found")),
    )
    assert ok is False
    assert msg == "no checkpoint found"


def test_cache_hit_skips_probe_and_rsync():
    def fail_run_on(*args, **kwargs):
        raise AssertionError("cache hit should not probe")

    ok, msg = resume_stage.stage_resume_ckpt_for_launch(
        _task(),
        "remote",
        source_loc={"node": "local", "path": "/ckpt/checkpoint.pt"},
        max_ckpt_mb=100,
        deps=_deps(
            cache_hits={("local", "remote", "/ckpt")},
            run_on_impl=fail_run_on,
        ),
    )

    assert ok is True
    assert msg == "cache hit (checkpoint already staged)"


def test_source_missing_size_probe_fail_and_cap_exceeded():
    def missing_source(node, cmd, **kwargs):
        if cmd.startswith("test -d"):
            return 1, "", ""
        return 0, "", ""

    ok, msg = resume_stage.stage_resume_ckpt_for_launch(
        _task(),
        "remote",
        source_loc={"node": "local", "path": "/ckpt/checkpoint.pt"},
        max_ckpt_mb=100,
        deps=_deps(run_on_impl=missing_source),
    )
    assert ok is False
    assert "missing on resume source local" in msg

    def bad_size(node, cmd, **kwargs):
        if cmd.startswith("test -d"):
            return 0, "", ""
        if "du -sm" in cmd:
            return 0, "not-a-number\n", ""
        return 0, "", ""

    ok, msg = resume_stage.stage_resume_ckpt_for_launch(
        _task(),
        "remote",
        source_loc={"node": "local", "path": "/ckpt/checkpoint.pt"},
        max_ckpt_mb=100,
        deps=_deps(run_on_impl=bad_size),
    )
    assert ok is False
    assert "size probe failed" in msg

    cap_marks = []

    def oversized(node, cmd, **kwargs):
        if cmd.startswith("test -d"):
            return 0, "", ""
        if "du -sm" in cmd:
            return 0, "101\n", ""
        return 0, "", ""

    ok, msg = resume_stage.stage_resume_ckpt_for_launch(
        _task(),
        "remote",
        source_loc={"node": "local", "path": "/ckpt/checkpoint.pt"},
        max_ckpt_mb=100,
        deps=_deps(cap_marks=cap_marks, run_on_impl=oversized),
    )
    assert ok is False
    assert msg.startswith("CAP_EXCEEDED:")
    assert cap_marks == [("local", "remote", "/ckpt")]


def test_windows_source_is_rejected_before_rsync():
    ok, msg = resume_stage.stage_resume_ckpt_for_launch(
        _task(),
        "remote",
        source_loc={"node": "win_src", "path": "/ckpt/checkpoint.pt"},
        max_ckpt_mb=100,
        deps=_deps(),
    )
    assert ok is False
    assert "staging from Windows source win_src is not supported yet" in msg


def test_remote_to_remote_relays_complete_directory_via_local_tmp():
    success_marks = []
    run_on_calls = []
    subprocess_calls = []

    ok, msg = resume_stage.stage_resume_ckpt_for_launch(
        _task(),
        "remote",
        source_loc={"node": "src_remote", "path": "/ckpt/checkpoint.pt"},
        max_ckpt_mb=100,
        deps=_deps(
            success_marks=success_marks,
            run_on_calls=run_on_calls,
            subprocess_calls=subprocess_calls,
        ),
    )

    assert ok is True
    assert msg == "synced ckpt_dir remote-to-remote via local (42MB)"
    assert len(subprocess_calls) == 2
    pull, push = subprocess_calls
    assert pull[:5] == [
        "rsync", "-az", "--partial", "-e", "ssh-shell-src_remote"]
    assert pull[-2] == "src_remote:/mnt/src_remote/ckpt/"
    assert "scheduleurm-ckpt-relay/" in pull[-1]
    assert push[:5] == [
        "rsync", "-az", "--partial", "-e", "ssh-shell-remote"]
    assert "scheduleurm-ckpt-relay/" in push[-2]
    assert push[-1] == "remote:/mnt/remote/ckpt/"
    assert any(
        call[0] == "remote" and call[1].startswith("test -e")
        for call in run_on_calls
    )
    assert success_marks == [("src_remote", "remote", "/ckpt")]


def test_remote_relay_keeps_partial_spool_and_reuses_it_on_retry(tmp_path):
    subprocess_calls = []
    success_marks = []
    retry_saw_partial = []

    def flaky_rsync(args, **kwargs):
        del kwargs
        call_index = len(subprocess_calls)
        if call_index == 1:
            spool = Path(args[-1])
            spool.mkdir(parents=True, exist_ok=True)
            (spool / "replay_buffer.npz").write_bytes(b"partial")
            return _RunResult(returncode=12, stderr="connection closed")
        if call_index == 2:
            retry_saw_partial.append(
                (Path(args[-1]) / "replay_buffer.npz").exists())
        return _RunResult()

    deps = _deps(
        success_marks=success_marks,
        subprocess_calls=subprocess_calls,
        subprocess_impl=flaky_rsync,
        relay_spool_root=str(tmp_path / "relay"),
    )
    source = {"node": "src_remote", "path": "/ckpt/checkpoint.pt"}

    ok, msg = resume_stage.stage_resume_ckpt_for_launch(
        _task(), "remote", source_loc=source, max_ckpt_mb=100, deps=deps)
    assert ok is False
    assert "connection closed" in msg
    spools = list((tmp_path / "relay").iterdir())
    assert len(spools) == 1
    assert (spools[0] / "replay_buffer.npz").exists()

    ok, msg = resume_stage.stage_resume_ckpt_for_launch(
        _task(), "remote", source_loc=source, max_ckpt_mb=100, deps=deps)
    assert ok is True
    assert retry_saw_partial == [True]
    assert success_marks == [("src_remote", "remote", "/ckpt")]
    assert list((tmp_path / "relay").iterdir()) == []


def test_windows_target_uses_windows_stager_and_marks_success():
    success_marks = []
    stage_calls = []

    ok, msg = resume_stage.stage_resume_ckpt_for_launch(
        _task(),
        "win",
        source_loc={"node": "local", "path": "/ckpt/checkpoint.pt"},
        max_ckpt_mb=100,
        deps=_deps(success_marks=success_marks, windows_stage_calls=stage_calls),
    )

    assert ok is True
    assert msg == "synced ckpt_dir to Windows (42MB): windows copied"
    assert stage_calls == [("/ckpt", "win", "/ckpt", {"timeout_s": 600})]
    assert success_marks == [("local", "win", "/ckpt")]


def test_relay_target_syncs_local_to_relay_then_relay_to_target():
    success_marks = []
    run_on_calls = []
    subprocess_calls = []

    ok, msg = resume_stage.stage_resume_ckpt_for_launch(
        _task(),
        "hpc",
        source_loc={"node": "local", "path": "/ckpt/checkpoint.pt"},
        max_ckpt_mb=100,
        deps=_deps(
            relay_node="gpu2",
            success_marks=success_marks,
            run_on_calls=run_on_calls,
            subprocess_calls=subprocess_calls,
        ),
    )

    assert ok is True
    assert msg == "synced ckpt_dir via gpu2 (42MB)"
    assert subprocess_calls[0][:4] == ["rsync", "-az", "--partial", "-e"]
    assert subprocess_calls[0][4] == "ssh-shell-gpu2"
    assert subprocess_calls[0][-2:] == ["/ckpt/", "gpu2:/relay/hpc/mnt/hpc/ckpt/"]
    assert any(call[0] == "gpu2" and "rsync" in call[1] for call in run_on_calls)
    assert success_marks == [("local", "hpc", "/ckpt")]


def test_direct_rsync_uses_pair_shell_verifies_and_marks_success():
    success_marks = []
    run_on_calls = []
    subprocess_calls = []

    ok, msg = resume_stage.stage_resume_ckpt_for_launch(
        _task(),
        "remote",
        source_loc={"node": "local", "path": "/ckpt/checkpoint.pt"},
        max_ckpt_mb=100,
        deps=_deps(
            success_marks=success_marks,
            run_on_calls=run_on_calls,
            subprocess_calls=subprocess_calls,
        ),
    )

    assert ok is True
    assert msg == "synced ckpt_dir (42MB)"
    assert subprocess_calls[0] == [
        "rsync",
        "-az",
        "--partial",
        "-e",
        "pair-shell-local-remote",
        "/ckpt/",
        "remote:/mnt/remote/ckpt/",
    ]
    assert any(call[0] == "remote" and call[1].startswith("test -e") for call in run_on_calls)
    assert success_marks == [("local", "remote", "/ckpt")]


def test_direct_rsync_failure_and_timeout_match_legacy_messages():
    def fail_rsync(args, **kwargs):
        return _RunResult(returncode=12, stderr="connection refused")

    ok, msg = resume_stage.stage_resume_ckpt_for_launch(
        _task(),
        "remote",
        source_loc={"node": "local", "path": "/ckpt/checkpoint.pt"},
        max_ckpt_mb=100,
        deps=_deps(subprocess_impl=fail_rsync),
    )
    assert ok is False
    assert msg == "rsync ckpt failed rc=12: connection refused"

    def timeout_rsync(args, **kwargs):
        raise subprocess.TimeoutExpired(args, timeout=600)

    ok, msg = resume_stage.stage_resume_ckpt_for_launch(
        _task(),
        "remote",
        source_loc={"node": "local", "path": "/ckpt/checkpoint.pt"},
        max_ckpt_mb=100,
        deps=_deps(subprocess_impl=timeout_rsync),
    )
    assert ok is False
    assert msg == "rsync ckpt timeout (>600s)"


def test_direct_mkdir_failure_prevents_rsync():
    subprocess_calls = []

    def fail_target_mkdir(node, cmd, **kwargs):
        if cmd.startswith("test -d"):
            return 0, "", ""
        if "du -sm" in cmd:
            return 0, "42\n", ""
        if cmd.startswith("mkdir -p"):
            return 1, "", "parent missing"
        return 0, "", ""

    ok, msg = resume_stage.stage_resume_ckpt_for_launch(
        _task(),
        "remote",
        source_loc={"node": "local", "path": "/ckpt/checkpoint.pt"},
        max_ckpt_mb=100,
        deps=_deps(
            run_on_impl=fail_target_mkdir,
            subprocess_calls=subprocess_calls,
        ),
    )

    assert ok is False
    assert msg == "checkpoint target mkdir failed rc=1: parent missing"
    assert subprocess_calls == []
