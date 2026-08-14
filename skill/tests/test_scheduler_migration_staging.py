from __future__ import annotations

import subprocess

from skill.scheduler_migration import staging as migration_stage
from skill.scheduler_migration.staging import MigrationStagingDeps


class _RunResult:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _task(**overrides):
    task = {
        "id": "t1",
        "cwd": "/work",
        "preferred_node": "src",
        "cmd": "/abs/python train.py",
    }
    task.update(overrides)
    return task


def _deps(
    *,
    nodes: dict | None = None,
    cache_hits: set[tuple] | None = None,
    success_marks: list[tuple] | None = None,
    run_on_calls: list[tuple] | None = None,
    subprocess_calls: list[list[str]] | None = None,
    run_on_impl=None,
    subprocess_impl=None,
    cwd_size_mb: int = 10,
    ckpt_size_mb: int = 100,
):
    nodes = nodes or {
        "src": {"host": None},
        "tgt": {"host": "tgtbox"},
        "src_remote": {"host": "srcbox"},
        "tgt_remote": {"host": "tgtbox"},
    }
    cache_hits = cache_hits or set()
    success_marks = success_marks if success_marks is not None else []
    run_on_calls = run_on_calls if run_on_calls is not None else []
    subprocess_calls = subprocess_calls if subprocess_calls is not None else []

    def run_on(node, cmd, **kwargs):
        run_on_calls.append((node, cmd, kwargs))
        if run_on_impl:
            return run_on_impl(node, cmd, **kwargs)
        if "du -sm" in cmd and "--exclude=.git" in cmd:
            return 0, f"{cwd_size_mb}\n", ""
        if "du -sm /ckpt" in cmd:
            return 0, f"{ckpt_size_mb}\n", ""
        if "test -d /ckpt" in cmd:
            return 0, "", ""
        if "test -d /work" in cmd:
            return 0, "", ""
        if "ls -1 /ckpt" in cmd:
            return 0, "model.pt\n", ""
        if "test -x" in cmd:
            return 0, "", ""
        return 0, "", ""

    def run_subprocess(args, **kwargs):
        subprocess_calls.append(list(args))
        if subprocess_impl:
            return subprocess_impl(args, **kwargs)
        return _RunResult()

    return MigrationStagingDeps(
        node_configs=nodes,
        migration_max_cwd_size_mb=2048,
        staging_cache_hit=lambda key: key in cache_hits,
        mark_staging_success=lambda key: success_marks.append(key),
        remote_path_for_node=lambda node, path: path,
        rsync_path_for_node=lambda node, path: path if not nodes.get(node, {}).get("host") else f"{node}:{path}",
        rsync_shell_for_pair=lambda source, target: (
            f"shell-{source}-{target}"
            if nodes.get(source, {}).get("host") or nodes.get(target, {}).get("host")
            else None
        ),
        run_on=run_on,
        apply_node_cmd_rewrites=lambda node, cmd: cmd,
        run_subprocess=run_subprocess,
    )


def test_no_cwd_and_source_target_short_circuit():
    ok, msg = migration_stage.stage_for_migration(
        {"id": "t0", "preferred_node": "src"},
        "tgt",
        max_ckpt_mb=2048,
        deps=_deps(),
    )
    assert ok is False
    assert msg == "no cwd on task"

    ok, msg = migration_stage.stage_for_migration(
        _task(preferred_node="tgt"),
        "tgt",
        max_ckpt_mb=2048,
        deps=_deps(),
    )
    assert ok is True
    assert msg == "source==target; nothing to stage"


def test_cwd_cache_miss_always_rsyncs_and_marks_cache():
    success_marks = []
    subprocess_calls = []

    ok, msg = migration_stage.stage_for_migration(
        _task(cmd="python train.py"),
        "tgt",
        max_ckpt_mb=2048,
        deps=_deps(success_marks=success_marks, subprocess_calls=subprocess_calls),
    )

    assert ok is True
    assert msg == "staged (cwd)"
    assert subprocess_calls == [[
        "rsync",
        "-az",
        "--partial",
        "--exclude=.git",
        "--exclude=__pycache__",
        "--exclude=*.pyc",
        "-e",
        "shell-src-tgt",
        "/work/",
        "tgt:/work/",
    ]]
    assert success_marks == [("src", "tgt", "/work")]


def test_unsafe_cwd_is_rejected_even_if_old_cache_marker_exists():
    run_on_calls = []
    subprocess_calls = []

    ok, msg = migration_stage.stage_for_migration(
        _task(cwd="/tmp", cmd="python train.py"),
        "tgt",
        max_ckpt_mb=2048,
        deps=_deps(
            cache_hits={("src", "tgt", "/tmp")},
            run_on_calls=run_on_calls,
            subprocess_calls=subprocess_calls,
        ),
    )

    assert ok is False
    assert msg.startswith("UNSAFE_CWD:")
    assert run_on_calls == []
    assert subprocess_calls == []


def test_cwd_cache_hit_skips_rsync_but_still_checks_env():
    subprocess_calls = []
    run_on_calls = []

    ok, msg = migration_stage.stage_for_migration(
        _task(),
        "tgt",
        max_ckpt_mb=2048,
        deps=_deps(
            cache_hits={("src", "tgt", "/work")},
            run_on_calls=run_on_calls,
            subprocess_calls=subprocess_calls,
        ),
    )

    assert ok is True
    assert msg == "staged (cwd + env)"
    assert subprocess_calls == []
    assert any(node == "tgt" and "test -x /abs/python" in cmd for node, cmd, _ in run_on_calls)


def test_cwd_cap_and_remote_to_remote_reject_before_rsync():
    subprocess_calls = []
    big = 4096

    ok, msg = migration_stage.stage_for_migration(
        _task(),
        "tgt",
        max_ckpt_mb=2048,
        deps=_deps(cwd_size_mb=big, subprocess_calls=subprocess_calls),
    )

    assert ok is False
    assert f"cwd /work is {big}MB > max 2048MB" in msg
    assert subprocess_calls == []

    ok, msg = migration_stage.stage_for_migration(
        _task(preferred_node="src_remote"),
        "tgt_remote",
        max_ckpt_mb=2048,
        deps=_deps(subprocess_calls=subprocess_calls),
    )

    assert ok is False
    assert "cwd rsync remote→remote not yet supported" in msg


def test_checkpoint_absent_is_first_launch_and_du_failure_is_fail_closed():
    def absent_ckpt(node, cmd, **kwargs):
        if "du -sm" in cmd and "--exclude=.git" in cmd:
            return 0, "10\n", ""
        if "test -d /work" in cmd:
            return 0, "", ""
        if "test -d /ckpt" in cmd:
            return 1, "", ""
        if "test -x" in cmd:
            return 0, "", ""
        return 0, "", ""

    ok, msg = migration_stage.stage_for_migration(
        _task(ckpt_dir="/ckpt"),
        "tgt",
        max_ckpt_mb=2048,
        deps=_deps(run_on_impl=absent_ckpt),
    )
    assert ok is True
    assert msg == "staged (cwd + env)"

    def du_fails(node, cmd, **kwargs):
        if "du -sm" in cmd and "--exclude=.git" in cmd:
            return 0, "10\n", ""
        if "test -d /work" in cmd:
            return 0, "", ""
        if "test -d /ckpt" in cmd:
            return 0, "", ""
        if "du -sm /ckpt" in cmd:
            return 1, "", "permission denied"
        return 0, "", ""

    ok, msg = migration_stage.stage_for_migration(
        _task(ckpt_dir="/ckpt"),
        "tgt",
        max_ckpt_mb=2048,
        deps=_deps(run_on_impl=du_fails),
    )
    assert ok is False
    assert "fail-closed" in msg
    assert "step-0 restart" in msg


def test_checkpoint_cap_happy_path_and_empty_target_verification():
    ok, msg = migration_stage.stage_for_migration(
        _task(ckpt_dir="/ckpt"),
        "tgt",
        max_ckpt_mb=50,
        deps=_deps(ckpt_size_mb=100),
    )
    assert ok is False
    assert "ckpt_dir /ckpt is 100MB > max 50MB" in msg

    success_marks = []
    subprocess_calls = []
    ok, msg = migration_stage.stage_for_migration(
        _task(ckpt_dir="/ckpt"),
        "tgt",
        max_ckpt_mb=2048,
        deps=_deps(
            success_marks=success_marks,
            subprocess_calls=subprocess_calls,
            ckpt_size_mb=100,
        ),
    )
    assert ok is True
    assert msg == "staged (cwd + ckpt + env)"
    assert success_marks == [("src", "tgt", "/work"), ("src", "tgt", "/ckpt")]
    assert [call[0] for call in subprocess_calls] == ["rsync", "rsync"]

    def empty_target(node, cmd, **kwargs):
        if "du -sm" in cmd and "--exclude=.git" in cmd:
            return 0, "10\n", ""
        if "du -sm /ckpt" in cmd:
            return 0, "100\n", ""
        if "test -d /ckpt" in cmd or "test -d /work" in cmd:
            return 0, "", ""
        if "ls -1 /ckpt" in cmd:
            return 0, "", ""
        if "test -x" in cmd:
            return 0, "", ""
        return 0, "", ""

    ok, msg = migration_stage.stage_for_migration(
        _task(ckpt_dir="/ckpt"),
        "tgt",
        max_ckpt_mb=2048,
        deps=_deps(run_on_impl=empty_target),
    )
    assert ok is False
    assert "appears empty" in msg


def test_env_missing_and_rsync_failure_messages_match_legacy():
    def env_missing(node, cmd, **kwargs):
        if "du -sm" in cmd and "--exclude=.git" in cmd:
            return 0, "10\n", ""
        if "test -d /work" in cmd:
            return 0, "", ""
        if "test -x /env/python" in cmd:
            return 1, "", ""
        return 0, "", ""

    ok, msg = migration_stage.stage_for_migration(
        _task(cmd="/env/python train.py"),
        "tgt",
        max_ckpt_mb=2048,
        deps=_deps(run_on_impl=env_missing),
    )
    assert ok is False
    assert msg == "python at /env/python not executable on tgt; deploy the conda env first (env_spec=conda:... if available)"

    def rsync_fails(args, **kwargs):
        return _RunResult(returncode=23, stderr="disk full")

    ok, msg = migration_stage.stage_for_migration(
        _task(cmd="python train.py"),
        "tgt",
        max_ckpt_mb=2048,
        deps=_deps(subprocess_impl=rsync_fails),
    )
    assert ok is False
    assert msg == "rsync cwd failed rc=23: disk full"

    def rsync_times_out(args, **kwargs):
        raise subprocess.TimeoutExpired(args, timeout=600)

    ok, msg = migration_stage.stage_for_migration(
        _task(cmd="python train.py"),
        "tgt",
        max_ckpt_mb=2048,
        deps=_deps(subprocess_impl=rsync_times_out),
    )
    assert ok is False
    assert msg == "rsync cwd timeout (>10min)"


def run(check, _sch):
    for fn in (
        test_no_cwd_and_source_target_short_circuit,
        test_cwd_cache_miss_always_rsyncs_and_marks_cache,
        test_unsafe_cwd_is_rejected_even_if_old_cache_marker_exists,
        test_cwd_cache_hit_skips_rsync_but_still_checks_env,
        test_cwd_cap_and_remote_to_remote_reject_before_rsync,
        test_checkpoint_absent_is_first_launch_and_du_failure_is_fail_closed,
        test_checkpoint_cap_happy_path_and_empty_target_verification,
        test_env_missing_and_rsync_failure_messages_match_legacy,
    ):
        try:
            fn()
            check(f"migration staging module: {fn.__name__}", True)
        except Exception as exc:
            check(f"migration staging module: {fn.__name__}", False, diag=repr(exc))
