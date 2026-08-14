from __future__ import annotations

from dataclasses import replace

from skill import scheduler_launch_cwd_staging as cwd_stage
from skill.scheduler_launch_cwd_staging import (
    CodeTarStagingDeps,
    LaunchCwdStagingDeps,
    WindowsTarStagingDeps,
)


class _RunResult:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _Pipe:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _FakePopen:
    def __init__(self, args, stdout=None, stderr=None, returncode=0, tar_err=b""):
        self.args = list(args)
        self.stdout = _Pipe()
        self.returncode = returncode
        self._tar_err = tar_err
        self.killed = False

    def communicate(self, timeout=None):
        return b"", self._tar_err

    def kill(self):
        self.killed = True


def _deps(
    *,
    nodes: dict | None = None,
    cache_hits: set[tuple] | None = None,
    success_marks: list[tuple] | None = None,
    cap_marks: list[tuple] | None = None,
    run_on_calls: list[tuple] | None = None,
    windows_result: tuple[int, str, str] = (0, "OK", ""),
    windows_stage_result: tuple[bool, str] = (True, "windows synced"),
    relay_node: str | None = None,
    code_tar_calls: list[tuple] | None = None,
):
    nodes = nodes or {
        "local": {"host": None},
        "remote": {"host": "user@remote"},
        "win": {"host": "winhost", "windows": True},
        "hpc": {"host": "hpc-host"},
        "gpu2": {"host": "gpu2-host"},
    }
    cache_hits = cache_hits or set()
    success_marks = success_marks if success_marks is not None else []
    cap_marks = cap_marks if cap_marks is not None else []
    run_on_calls = run_on_calls if run_on_calls is not None else []
    code_tar_calls = code_tar_calls if code_tar_calls is not None else []

    def run_on(node, cmd, **kwargs):
        run_on_calls.append((node, cmd, kwargs))
        return 0, "", ""

    def code_tar(task, target_node, remote_cwd, cwd_size_mb):
        code_tar_calls.append((task, target_node, remote_cwd, cwd_size_mb))
        return True, f"code_tar {cwd_size_mb}"

    return LaunchCwdStagingDeps(
        node_configs=nodes,
        launch_max_cwd_size_mb=2048,
        staging_cache_hit=lambda key: key in cache_hits,
        mark_stage_success=lambda key: success_marks.append(key),
        mark_cap_exceeded=lambda key: cap_marks.append(key),
        node_is_windows=lambda node: bool(nodes.get(node, {}).get("windows")),
        is_windows_native_path=lambda path: str(path).startswith("C:\\"),
        windows_path_for_node=lambda node, path: f"C:\\stage{path}",
        run_windows_ps=lambda node, script, **kwargs: windows_result,
        ps_quote=lambda text: f"'{text}'",
        stage_local_dir_to_windows=lambda *args, **kwargs: windows_stage_result,
        remote_path_for_node=lambda node, path: f"/remote/{node}{path}",
        run_on=run_on,
        stage_code_tar_for_launch=code_tar,
        relay_node_for_node=lambda node: relay_node if node == "hpc" else None,
        relay_path_for_node=lambda node, path: f"/relay/{node}{path}",
        rsync_path_for_node=lambda node, path: f"{node}:{path}",
        ssh_rsync_shell_for_node=lambda node: f"ssh-shell-{node}",
        relay_ssh_target_for_node=lambda node: f"{node}-via-relay",
        run_control_subprocess=cwd_stage.subprocess.run,
    )


def test_stage_cwd_local_target_short_circuits(monkeypatch):
    monkeypatch.setattr(
        cwd_stage.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no subprocess")),
    )

    ok, msg = cwd_stage.stage_cwd_for_launch(
        {"cwd": "/tmp"},
        "local",
        deps=_deps(),
    )

    assert ok is True
    assert "nothing to sync" in msg


def test_stage_cwd_cache_hit_skips_path_and_rsync(monkeypatch):
    monkeypatch.setattr(
        cwd_stage.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no subprocess")),
    )

    ok, msg = cwd_stage.stage_cwd_for_launch(
        {"cwd": "/does/not/need/to/exist"},
        "remote",
        deps=_deps(cache_hits={("local", "remote", "/does/not/need/to/exist")}),
    )

    assert ok is True
    assert "cache hit" in msg


def test_launch_input_state_infers_wait_file_parent_for_legacy_task(tmp_path):
    cwd = tmp_path / "project"
    bundle = cwd / "bundles" / "seed_0" / "sac"
    bundle.mkdir(parents=True)
    manifest = bundle / "bundle_manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    task = {"cwd": str(cwd), "wait_for_files": [str(manifest)]}
    cache_hits = {("sentinel",)}
    deps = _deps(cache_hits=cache_hits)

    assert cwd_stage.launch_input_stage_state(task, "remote", deps=deps) == "needs_stage"
    cache_hits.add(cwd_stage.launch_input_stage_key(task, "remote"))
    assert cwd_stage.launch_input_stage_state(task, "remote", deps=deps) == "ready"


def test_launch_input_key_aliases_shared_workspace_nodes(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    task = {"stage_input_paths": [str(bundle)]}
    nodes = {
        "node001": {"shared_workspace_group": "hpc-home"},
        "node002": {"shared_workspace_group": "hpc-home"},
    }

    assert cwd_stage.launch_input_stage_key(
        task, "node001", node_configs=nodes
    ) == cwd_stage.launch_input_stage_key(
        task, "node002", node_configs=nodes
    )


def test_launch_input_sync_keeps_bundle_logs(monkeypatch, tmp_path):
    bundle = tmp_path / "project" / "bundle"
    logs = bundle / "logs"
    logs.mkdir(parents=True)
    (logs / "protocol_signature.json").write_text("{}\n", encoding="utf-8")
    calls = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        if args[0] == "du":
            return _RunResult(stdout="1\t.\n")
        return _RunResult()

    monkeypatch.setattr(cwd_stage.subprocess, "run", fake_run)
    task = {"stage_input_paths": [str(bundle)]}
    ok, _ = cwd_stage.stage_input_paths_for_launch(task, "remote", deps=_deps())

    assert ok is True
    rsync_call = next(args for args in calls if args[0] == "rsync")
    assert "--exclude=logs/" not in rsync_call


def test_launch_input_set_is_marked_only_after_every_directory(monkeypatch, tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    calls = []
    success_marks = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        if args[0] == "du":
            return _RunResult(stdout="1\t.\n")
        return _RunResult()

    monkeypatch.setattr(cwd_stage.subprocess, "run", fake_run)
    task = {"stage_input_paths": [str(first), str(second)]}
    key = cwd_stage.launch_input_stage_key(task, "remote")

    ok, _ = cwd_stage.stage_input_paths_for_launch(
        task, "remote", deps=_deps(success_marks=success_marks))

    assert ok is True
    assert len([args for args in calls if args[0] == "rsync"]) == 2
    assert success_marks == [key]


def test_launch_input_set_failure_does_not_publish_cache(monkeypatch, tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    rsync_count = 0
    success_marks = []

    def fake_run(args, **kwargs):
        nonlocal rsync_count
        if args[0] == "du":
            return _RunResult(stdout="1\t.\n")
        rsync_count += 1
        return _RunResult(returncode=1, stderr="transfer failed") \
            if rsync_count == 2 else _RunResult()

    monkeypatch.setattr(cwd_stage.subprocess, "run", fake_run)
    task = {"stage_input_paths": [str(first), str(second)]}

    ok, msg = cwd_stage.stage_input_paths_for_launch(
        task, "remote", deps=_deps(success_marks=success_marks))

    assert ok is False
    assert "failed" in msg
    assert success_marks == []


def test_stage_cwd_missing_source_fails_clearly():
    ok, msg = cwd_stage.stage_cwd_for_launch(
        {"cwd": "/nonexistent_scheduleurm_cwd_staging_test"},
        "remote",
        deps=_deps(),
    )

    assert ok is False
    assert "can't seed target" in msg


def test_stage_cwd_cap_exceeded_marks_cap(monkeypatch, tmp_path):
    cap_marks = []

    def fake_run(args, **kwargs):
        assert args[0] == "du"
        assert "--exclude=outputs/run1" in args
        return _RunResult(stdout="9\t.\n")

    monkeypatch.setattr(cwd_stage.subprocess, "run", fake_run)

    deps = replace(
        _deps(cap_marks=cap_marks),
        launch_max_cwd_size_mb=1,
    )
    ok, msg = cwd_stage.stage_cwd_for_launch(
        {"cwd": str(tmp_path)},
        "remote",
        extra_excludes=["outputs/run1/"],
        deps=deps,
    )

    assert ok is False
    assert msg.startswith("CAP_EXCEEDED:")
    assert cap_marks == [("local", "remote", str(tmp_path))]


def test_stage_cwd_windows_native_path_probes_target_and_marks_success():
    success_marks = []

    ok, msg = cwd_stage.stage_cwd_for_launch(
        {"cwd": "C:\\work\\proj"},
        "win",
        deps=_deps(success_marks=success_marks),
    )

    assert ok is True
    assert "target-native Windows cwd exists" in msg
    assert success_marks == [("local", "win", "C:\\work\\proj")]


def test_stage_cwd_windows_copy_uses_windows_stager(monkeypatch, tmp_path):
    success_marks = []
    monkeypatch.setattr(
        cwd_stage.subprocess,
        "run",
        lambda args, **kwargs: _RunResult(stdout="2\t.\n"),
    )

    ok, msg = cwd_stage.stage_cwd_for_launch(
        {"cwd": str(tmp_path)},
        "win",
        deps=_deps(success_marks=success_marks),
    )

    assert ok is True
    assert "windows synced" in msg
    assert "(2MB)" in msg
    assert success_marks == [("local", "win", str(tmp_path))]


def test_stage_cwd_direct_rsync_uses_delete_excludes_and_marks_success(monkeypatch, tmp_path):
    success_marks = []
    run_on_calls = []
    subprocess_calls = []

    def fake_run(args, **kwargs):
        subprocess_calls.append(list(args))
        if args[0] == "du":
            return _RunResult(stdout="3\t.\n")
        return _RunResult()

    monkeypatch.setattr(cwd_stage.subprocess, "run", fake_run)

    ok, msg = cwd_stage.stage_cwd_for_launch(
        {"cwd": str(tmp_path)},
        "remote",
        extra_excludes=["runs/seed1/", "outputs/seed2/"],
        deps=_deps(success_marks=success_marks, run_on_calls=run_on_calls),
    )

    assert ok is True
    assert msg == "synced (3MB)"
    rsync_call = next(args for args in subprocess_calls if args[0] == "rsync")
    assert "--delete" in rsync_call
    assert "--exclude=results/" in rsync_call
    assert "--exclude=runs/seed1/" in rsync_call
    assert "--exclude=outputs/seed2/" in rsync_call
    assert ["-e", "ssh-shell-remote"] == rsync_call[rsync_call.index("-e"):rsync_call.index("-e") + 2]
    assert success_marks == [("local", "remote", str(tmp_path))]
    assert run_on_calls[0][0] == "remote"
    assert "mkdir -p" in run_on_calls[0][1]


def test_stage_cwd_direct_rsync_uses_injected_control_runner(monkeypatch, tmp_path):
    control_calls = []

    def fake_run(args, **kwargs):
        if args[0] == "du":
            return _RunResult(stdout="1\t.\n")
        raise AssertionError("rsync must use injected control runner")

    def control_runner(args, **kwargs):
        control_calls.append((list(args), kwargs))
        return _RunResult()

    monkeypatch.setattr(cwd_stage.subprocess, "run", fake_run)

    ok, msg = cwd_stage.stage_cwd_for_launch(
        {"cwd": str(tmp_path)},
        "remote",
        deps=replace(_deps(), run_control_subprocess=control_runner),
    )

    assert ok is True
    assert msg == "synced (1MB)"
    assert control_calls and control_calls[0][0][0] == "rsync"
    assert control_calls[0][1]["timeout"] == 600


def test_stage_cwd_relay_syncs_local_to_relay_then_relay_to_target(monkeypatch, tmp_path):
    success_marks = []
    run_on_calls = []
    subprocess_calls = []

    def fake_run(args, **kwargs):
        subprocess_calls.append(list(args))
        if args[0] == "du":
            return _RunResult(stdout="4\t.\n")
        return _RunResult()

    monkeypatch.setattr(cwd_stage.subprocess, "run", fake_run)

    ok, msg = cwd_stage.stage_cwd_for_launch(
        {"cwd": str(tmp_path)},
        "hpc",
        deps=_deps(
            success_marks=success_marks,
            run_on_calls=run_on_calls,
            relay_node="gpu2",
        ),
    )

    assert ok is True
    assert msg == "synced via gpu2 (4MB)"
    assert any(args[0] == "rsync" and "gpu2:" in args[-1] for args in subprocess_calls)
    relay_cmds = [cmd for node, cmd, _ in run_on_calls if node == "gpu2" and "rsync" in cmd]
    assert relay_cmds
    assert "hpc-via-relay:" in relay_cmds[0]
    assert success_marks == [("local", "hpc", str(tmp_path))]


def test_stage_cwd_code_tar_delegates_after_size_probe(monkeypatch, tmp_path):
    code_tar_calls = []

    monkeypatch.setattr(
        cwd_stage.subprocess,
        "run",
        lambda args, **kwargs: _RunResult(stdout="5\t.\n"),
    )

    ok, msg = cwd_stage.stage_cwd_for_launch(
        {"cwd": str(tmp_path), "id": "t-code"},
        "remote",
        deps=_deps(
            nodes={
                "local": {"host": None},
                "remote": {
                    "host": "user@remote",
                    "launch_staging_method": "code_tar",
                },
            },
            code_tar_calls=code_tar_calls,
        ),
    )

    assert ok is True
    assert msg == "code_tar 5"
    assert code_tar_calls == [
        (
            {"cwd": str(tmp_path), "id": "t-code"},
            "remote",
            f"/remote/remote{tmp_path}",
            5,
        )
    ]


def test_stage_local_dir_to_windows_streams_tar_over_ssh(tmp_path):
    calls = {"popen": [], "run": [], "ps": []}

    def popen(args, **kwargs):
        calls["popen"].append((list(args), kwargs))
        return _FakePopen(args)

    def run(args, **kwargs):
        calls["run"].append((list(args), kwargs))
        return _RunResult(returncode=0, stdout=b"READY\n", stderr=b"")

    deps = WindowsTarStagingDeps(
        node_is_windows=lambda node: node == "win",
        windows_path_for_node=lambda node, path: f"C:\\stage{path}",
        ps_quote=lambda text: f"'{text}'",
        ssh_base_args=lambda node: ["ssh", "user@win"],
        run_windows_ps=lambda node, script, **kwargs: calls["ps"].append((node, script, kwargs)) or (0, "OK", ""),
        popen=popen,
        run_subprocess=run,
    )

    ok, msg = cwd_stage.stage_local_dir_to_windows(
        str(tmp_path),
        "win",
        "/work/proj",
        extra_excludes=["outputs/run1"],
        timeout_s=10,
        deps=deps,
    )

    assert ok is True
    assert msg == "synced to Windows C:\\stage/work/proj"
    tar_args = calls["popen"][0][0]
    assert tar_args[:4] == ["tar", "-h", "-C", str(tmp_path)]
    assert "--exclude=outputs/run1" in tar_args
    assert "--exclude=./outputs/run1" in tar_args
    ssh_args = calls["run"][0][0]
    assert ssh_args[:2] == ["ssh", "user@win"]
    assert "-EncodedCommand" in ssh_args
    assert calls["ps"] and "Test-Path" in calls["ps"][0][1]


def test_stage_local_dir_to_windows_reports_extract_failure(tmp_path):
    deps = WindowsTarStagingDeps(
        node_is_windows=lambda node: True,
        windows_path_for_node=lambda node, path: path,
        ps_quote=lambda text: repr(text),
        ssh_base_args=lambda node: ["ssh", "target"],
        run_windows_ps=lambda *args, **kwargs: (0, "OK", ""),
        popen=lambda args, **kwargs: _FakePopen(args),
        run_subprocess=lambda args, **kwargs: _RunResult(returncode=12, stdout=b"", stderr=b"bad extract"),
    )

    ok, msg = cwd_stage.stage_local_dir_to_windows(str(tmp_path), "win", "/x", deps=deps)

    assert ok is False
    assert "Windows extract failed rc=12" in msg


def test_windows_extract_error_is_not_hidden_by_local_tar_sigpipe(tmp_path):
    deps = WindowsTarStagingDeps(
        node_is_windows=lambda node: True,
        windows_path_for_node=lambda node, path: path,
        ps_quote=lambda text: repr(text),
        ssh_base_args=lambda node: ["ssh", "target"],
        run_windows_ps=lambda *args, **kwargs: (0, "OK", ""),
        popen=lambda args, **kwargs: _FakePopen(args, returncode=-13),
        run_subprocess=lambda args, **kwargs: _RunResult(
            returncode=12, stdout=b"", stderr=b"remote mkdir denied"),
    )

    ok, msg = cwd_stage.stage_local_dir_to_windows(
        str(tmp_path), "win", "/x", deps=deps)

    assert ok is False
    assert msg == "Windows extract failed rc=12: remote mkdir denied"


def test_stage_code_tar_for_launch_creates_scp_unpacks_and_marks_success(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print(1)", encoding="utf-8")
    (tmp_path / "configs").mkdir()
    run_calls = []
    subprocess_calls = []
    marks = []

    def run_subprocess(args, **kwargs):
        subprocess_calls.append((list(args), kwargs))
        return _RunResult(returncode=0)

    deps = CodeTarStagingDeps(
        node_configs={"node001": {"launch_stage_include_paths": ["src", "configs", "missing"]}},
        ssh_base_args=lambda node: ["ssh", "-p", "2222", "user@node001"],
        run_on=lambda node, cmd, **kwargs: run_calls.append((node, cmd, kwargs)) or (0, "", ""),
        mark_stage_success=lambda key: marks.append(key),
        getpid=lambda: 111,
        now=lambda: 222.0,
        run_subprocess=run_subprocess,
        tmp_dir=str(tmp_path),
    )

    ok, msg = cwd_stage.stage_code_tar_for_launch(
        {
            "id": "t1",
            "cwd": str(tmp_path),
            "stage_excludes": ["src/checkpoints", "src/results"],
        },
        "node001",
        "/remote/work",
        5,
        deps=deps,
    )

    assert ok is True
    assert msg == "code_tar staged 2 paths to node001 (5MB cwd)"
    assert subprocess_calls[0][0][:2] == ["tar", "-cf"]
    assert subprocess_calls[0][0][-2:] == ["src", "configs"]
    assert "--exclude=src/checkpoints" in subprocess_calls[0][0]
    assert "--exclude=src/results/**" in subprocess_calls[0][0]
    assert subprocess_calls[0][1]["cwd"] == str(tmp_path)
    assert subprocess_calls[1][0][:3] == ["scp", "-p", "2222"]
    assert subprocess_calls[1][0][-1] == (
        "user@node001:/remote/work/"
        ".scheduleurm_code_stage_111_222000000000_t1.tar"
    )
    assert run_calls and "tar -xf" in run_calls[0][1]
    assert "rm -rf" not in run_calls[0][1]
    assert marks == [("local", "node001", str(tmp_path))]


def test_stage_code_tar_for_launch_rejects_unsafe_include(tmp_path):
    deps = CodeTarStagingDeps(
        node_configs={"node001": {"launch_stage_include_paths": ["../secrets"]}},
        ssh_base_args=lambda node: ["ssh", "node"],
        run_on=lambda *args, **kwargs: (0, "", ""),
        mark_stage_success=lambda key: None,
    )

    ok, msg = cwd_stage.stage_code_tar_for_launch(
        {"cwd": str(tmp_path)},
        "node001",
        "/remote",
        1,
        deps=deps,
    )

    assert ok is False
    assert "unsafe code_tar include path" in msg
