from __future__ import annotations

from skill.scheduler_environment.env_smoke import EnvSmokeDeps, smoke_test_envs, smoke_test_envs_enabled


def _deps(tasks, *, run_results=None, windows=None):
    calls = {"run": [], "notify": []}
    run_results = list(run_results or [])
    windows = set(windows or [])

    def run_on(node, cmd, **kwargs):
        calls["run"].append((node, cmd, kwargs))
        if run_results:
            return run_results.pop(0)
        return 0, "ok\n", ""

    deps = EnvSmokeDeps(
        load_state=lambda: {"tasks": tasks},
        notify=lambda event, payload, **kwargs: calls["notify"].append((event, payload, kwargs)),
        node_is_windows=lambda node: node in windows,
        apply_node_cmd_rewrites=lambda node, cmd: cmd.replace("$ROOT", "/repo"),
        run_on=run_on,
        environ={},
    )
    return deps, calls


def test_smoke_test_envs_probes_absolute_python_and_conda_once():
    tasks = [
        {
            "id": "abs",
            "status": "queued",
            "node": "node001",
            "cmd": "/envs/a/bin/python train.py",
        },
        {
            "id": "dup",
            "status": "running",
            "node": "node001",
            "cmd": "/envs/a/bin/python eval.py",
        },
        {
            "id": "conda",
            "status": "launching",
            "require_node": "node002",
            "cmd": "conda run --no-capture-output -n envB python train.py",
        },
    ]
    deps, calls = _deps(tasks)

    smoke_test_envs(deps=deps)

    assert calls["run"] == [
        ("node001", "/envs/a/bin/python -c 'print(\"ok\")'", {"timeout": 15, "check": False}),
        ("node002", "conda run -n envB python -c 'print(\"ok\")' 2>&1", {"timeout": 15, "check": False}),
    ]
    assert calls["notify"] == [
        ("env_smoke_test_passed", {"checked": 2}, {"feishu_enabled": False})
    ]


def test_smoke_test_envs_skips_docker_adopted_nonactive_and_windows_tasks():
    tasks = [
        {"id": "done", "status": "done", "node": "node001", "cmd": "/x/bin/python a.py"},
        {"id": "docker", "status": "queued", "node": "node001", "env_spec": "docker:img", "cmd": "/x/bin/python a.py"},
        {"id": "adopted", "status": "running", "node": "node001", "auto_adopted": True, "cmd": "/x/bin/python a.py"},
        {"id": "win", "status": "queued", "node": "win", "cmd": "/x/bin/python a.py"},
    ]
    deps, calls = _deps(tasks, windows={"win"})

    smoke_test_envs(deps=deps)

    assert calls["run"] == []
    assert calls["notify"] == [
        ("env_smoke_test_passed", {"checked": 0}, {"feishu_enabled": False})
    ]


def test_smoke_test_envs_reports_probe_failures_and_load_errors():
    deps, calls = _deps(
        [{"id": "bad", "status": "queued", "node": "node001", "cmd": "/bad/bin/python a.py"}],
        run_results=[(1, "", "missing python")],
    )

    smoke_test_envs(deps=deps)

    assert calls["notify"] == [
        (
            "env_smoke_test_failed",
            {"count": 1, "details": [{"node": "node001", "env": "/bad/bin/python", "err": "missing python"}]},
            {},
        )
    ]

    error_calls = []
    error_deps = EnvSmokeDeps(
        load_state=lambda: (_ for _ in ()).throw(RuntimeError("state broken")),
        notify=lambda event, payload, **kwargs: error_calls.append((event, payload, kwargs)),
        node_is_windows=lambda node: False,
        apply_node_cmd_rewrites=lambda node, cmd: cmd,
        run_on=lambda *args, **kwargs: (0, "", ""),
        environ={},
    )
    smoke_test_envs(deps=error_deps)
    assert error_calls == [
        ("env_smoke_test_error", {"error": "could not load state: state broken"}, {"feishu_enabled": False})
    ]


def test_smoke_test_envs_enabled_parses_env_flags():
    assert smoke_test_envs_enabled({"SCHEDULEURM_ENV_SMOKE_TEST": "1"}) is True
    assert smoke_test_envs_enabled({"SCHEDULEURM_ENV_SMOKE_TEST": "true"}) is True
    assert smoke_test_envs_enabled({"SCHEDULEURM_ENV_SMOKE_TEST": "off"}) is False
    assert smoke_test_envs_enabled({}) is False
