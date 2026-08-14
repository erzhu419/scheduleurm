from __future__ import annotations

import json
import re

from skill.scheduler_migration.resume_scan import (
    ResumeScanDeps,
    allow_initial_resume_scan_error,
    find_resume_on_node,
    refresh_resume_locations_for_task,
    resume_candidate_is_safe,
    resume_node_names,
    scan_resume_locations,
    task_requires_resume_scan,
)


SAFE_RE = re.compile(r"(checkpoint|ckpt|resume|state|snapshot|epoch|step|iter|iteration)", re.I)
UNSAFE_RE = re.compile(
    r"(^|[_\-.])(model_?final|final_?model|final|best|buffer|replay|rollout|metrics?|results?|eval|train_?log)([_\-.]|$)",
    re.I,
)


def _deps(*, calls=None, run_on=None, run_windows_ps=None, now=1000.0, workers=1):
    calls = calls if calls is not None else []

    def default_run_on(node, cmd, **kwargs):
        calls.append(("run_on", node, cmd, kwargs))
        return 0, "", ""

    def default_windows(node, ps, **kwargs):
        calls.append(("run_windows_ps", node, ps, kwargs))
        return 0, "", ""

    def windows_path_for_node(node, path):
        return "F:\\mapped" + path.replace("/", "\\")

    return ResumeScanDeps(
        node_configs={
            "local": {"host": None, "capabilities": ["cuda"]},
            "node001": {"host": "h1", "capabilities": ["cuda"]},
            "node002": {"host": "h2", "capabilities": ["resac"]},
            "node003": {"host": "h3", "max_vram_per_task": 0},
            "win": {"host": "w", "os": "windows"},
        },
        ckpt_exts=("pt", "pth", "pkl", "ckpt", "bin", "safetensors", "npy", "npz", "h5", "hdf5", "tar"),
        resume_safe_name_re=SAFE_RE,
        resume_unsafe_name_re=UNSAFE_RE,
        resume_scan_ttl_s=120,
        resume_scan_workers=workers,
        task_gpu_capability=lambda task: task.get("capability") or "cuda",
        node_is_windows=lambda node: node == "win",
        windows_path_for_node=windows_path_for_node,
        ps_quote=lambda text: "'" + text.replace("'", "''") + "'",
        run_windows_ps=run_windows_ps or default_windows,
        remote_path_for_node=lambda node, path: path if node == "local" else f"/remote/{node}{path}",
        run_on=run_on or default_run_on,
        cmd_has_resume_flag=lambda cmd: "--resume" in cmd,
        now=lambda: now,
    )


def test_resume_candidate_safety_defaults_and_explicit_glob():
    deps = _deps()

    assert resume_candidate_is_safe("/x/checkpoint_10.pt", deps=deps) is True
    assert resume_candidate_is_safe("/x/model_final.pt", deps=deps) is False
    assert resume_candidate_is_safe("/x/model_final.pt", explicit_glob=True, deps=deps) is True
    assert resume_candidate_is_safe("/x/checkpoint.txt", deps=deps) is False


def test_resume_node_names_respects_explicit_known_down_and_gpu_filters():
    nodes = [
        {"name": "local", "alive": True},
        {"name": "node001", "alive": True},
        {"name": "node002", "alive": True},
        {"name": "node003", "alive": True},
        {"name": "win", "alive": True},
    ]
    deps = _deps()

    assert resume_node_names(
        {"require_node": "node001", "allowed_nodes": ["node002"], "est_vram_mb": 0},
        nodes,
        deps=deps,
    ) == ["node001", "node002", "local"]

    task = {
        "resume_locations": [{"node": "node002", "path": "/ckpt"}],
        "preferred_node": "local",
        "est_vram_mb": 0,
    }
    assert resume_node_names(task, [{"name": "local", "alive": True}, {"name": "node002", "alive": False}], deps=deps) == [
        "node002",
        "local",
    ]

    gpu_task = {"est_vram_mb": 1000, "capability": "cuda"}
    assert resume_node_names(gpu_task, nodes, deps=deps) == ["local", "node001"]


def test_remote_find_resume_parses_shell_metadata_without_remote_python():
    calls = []

    def run_on(node, cmd, **kwargs):
        calls.append((node, cmd, kwargs))
        payload = "2.0\t10\t/ckpt/checkpoint_2.pt"
        return 0, "noise\n__SCHEDULEURM_RESUME_SCAN_JSON_BEGIN__\n" + payload + "\n__SCHEDULEURM_RESUME_SCAN_JSON_END__\nmore", ""

    deps = _deps(calls=calls, run_on=run_on)
    found = find_resume_on_node({"ckpt_dir": "/ckpt", "ckpt_glob": "*"}, "node001", deps=deps)

    assert found == {"path": "/ckpt/checkpoint_2.pt", "mtime": 2.0, "size": 10, "node": "node001"}
    assert " find " in calls[0][1]
    assert "python" not in calls[0][1]
    assert "cd /" in calls[0][1]
    assert "/remote/node001/ckpt" in calls[0][1]


def test_remote_find_resume_supports_nested_relative_glob():
    calls = []

    def run_on(node, cmd, **kwargs):
        calls.append((node, cmd, kwargs))
        return (
            0,
            "2.0\t453189\t/run/checkpoints/train_state.pkl\n",
            "",
        )

    found = find_resume_on_node(
        {"ckpt_dir": "/run", "ckpt_glob": "checkpoints/train_state.pkl"},
        "local",
        deps=_deps(calls=calls, run_on=run_on),
    )

    assert found == {
        "path": "/run/checkpoints/train_state.pkl",
        "mtime": 2.0,
        "size": 453189,
        "node": "local",
    }
    assert "-maxdepth 2" in calls[0][1]
    assert "-path /run/checkpoints/train_state.pkl" in calls[0][1]


def test_resume_scan_rejects_glob_outside_checkpoint_root():
    try:
        find_resume_on_node(
            {"ckpt_dir": "/run", "ckpt_glob": "../train_state.pkl"},
            "local",
            deps=_deps(),
        )
    except RuntimeError as exc:
        assert "must be relative" in str(exc)
    else:
        raise AssertionError("parent traversal glob should be rejected")


def test_remote_find_resume_reports_bad_json_and_nonzero_rc():
    deps_bad = _deps(run_on=lambda *args, **kwargs: (0, "not-json\n", ""))
    try:
        find_resume_on_node({"ckpt_dir": "/ckpt"}, "node001", deps=deps_bad)
    except RuntimeError as exc:
        assert "bad checkpoint scan output" in str(exc)
    else:
        raise AssertionError("bad json should raise")

    deps_rc = _deps(run_on=lambda *args, **kwargs: (7, "", "ssh failed"))
    try:
        find_resume_on_node({"ckpt_dir": "/ckpt"}, "node001", deps=deps_rc)
    except RuntimeError as exc:
        assert "ssh failed" in str(exc)
    else:
        raise AssertionError("nonzero rc should raise")


def test_remote_find_resume_keeps_legacy_json_adapter_compatibility():
    payload = json.dumps({
        "path": "/ckpt/checkpoint_7.pt",
        "mtime": 7,
        "size": 11,
    })
    deps = _deps(run_on=lambda *args, **kwargs: (0, payload, ""))

    found = find_resume_on_node(
        {"ckpt_dir": "/ckpt"}, "node001", deps=deps
    )

    assert found == {
        "path": "/ckpt/checkpoint_7.pt",
        "mtime": 7.0,
        "size": 11,
        "node": "node001",
    }


def test_windows_find_resume_maps_path_and_parses_output():
    calls = []

    def run_windows(node, ps, **kwargs):
        calls.append((node, ps, kwargs))
        return 0, json.dumps({"path": r"F:\ckpt\checkpoint.pt", "mtime": 3, "size": 4}), ""

    found = find_resume_on_node({"ckpt_dir": "/proj/ckpt", "ckpt_glob": "*.pt"}, "win", deps=_deps(calls=calls, run_windows_ps=run_windows))

    assert found == {"path": r"F:\ckpt\checkpoint.pt", "mtime": 3, "size": 4, "node": "win"}
    assert "F:\\mapped\\proj\\ckpt" in calls[0][1]
    assert "Get-ChildItem" in calls[0][1]


def test_windows_find_resume_supports_nested_relative_glob():
    calls = []

    def run_windows(node, ps, **kwargs):
        calls.append((node, ps, kwargs))
        return 0, json.dumps({
            "path": r"F:\\mapped\\run\\checkpoints\\train_state.pkl",
            "mtime": 3,
            "size": 4,
        }), ""

    found = find_resume_on_node(
        {"ckpt_dir": "/run", "ckpt_glob": "checkpoints/train_state.pkl"},
        "win",
        deps=_deps(calls=calls, run_windows_ps=run_windows),
    )

    assert found["node"] == "win"
    assert "Split-Path -Path $relative -Parent" in calls[0][1]
    assert "Join-Path -Path $dir -ChildPath $parent" in calls[0][1]


def test_scan_resume_locations_uses_cache_and_sorts_newest_first():
    calls = []

    def run_on(node, cmd, **kwargs):
        calls.append(node)
        if node == "local":
            return 0, "1.0\t1\t/ckpt/checkpoint_1.pt", ""
        if node == "node001":
            return 0, "3.0\t1\t/ckpt/checkpoint_3.pt", ""
        return 0, "", ""

    deps = _deps(calls=calls, run_on=run_on, workers=1)
    task = {"ckpt_dir": "/ckpt", "ckpt_glob": "*", "cmd": "python train.py --resume", "est_vram_mb": 0}
    cache = {}

    locs, errs = scan_resume_locations(task, nodes=[{"name": "local", "alive": True}, {"name": "node001", "alive": True}], cache=cache, deps=deps)
    locs2, errs2 = scan_resume_locations(task, nodes=[{"name": "local", "alive": True}, {"name": "node001", "alive": True}], cache=cache, deps=deps)

    assert [loc["node"] for loc in locs] == ["node001", "local"]
    assert errs == {}
    assert (locs2, errs2) == (locs, errs)
    assert calls == ["local", "node001"]


def test_refresh_resume_locations_ttl_and_cleanup():
    calls = []

    def run_on(node, cmd, **kwargs):
        calls.append(node)
        return 0, "1.0\t1\t/ckpt/checkpoint.pt", ""

    deps = _deps(calls=calls, run_on=run_on, now=1000.0)
    task = {"ckpt_dir": "/ckpt", "cmd": "python train.py --resume", "est_vram_mb": 0}
    nodes = [{"name": "local", "alive": True}]

    locs, errs = refresh_resume_locations_for_task(task, nodes, {}, deps=deps)
    locs2, errs2 = refresh_resume_locations_for_task(task, nodes, {}, deps=deps)

    assert locs == locs2
    assert errs == errs2 == {}
    assert calls == ["local"]
    assert task["resume_preferred_nodes"] == ["local"]
    assert task["resume_checkpoint_node"] == "local"
    assert task["resume_from"] == "/ckpt/checkpoint.pt"

    task["skip_resume_scan"] = True
    refresh_resume_locations_for_task(task, nodes, {}, deps=deps)
    assert "resume_locations" not in task
    assert task["resume_from"] == "/ckpt/checkpoint.pt"


def test_task_requires_and_initial_scan_error_gates():
    deps = _deps()

    assert task_requires_resume_scan({"ckpt_dir": "/c", "cmd": "python train.py --resume"}, deps=deps) is True
    assert task_requires_resume_scan({"ckpt_dir": "/c", "skip_resume_scan": True, "cmd": "--resume"}, deps=deps) is False
    assert task_requires_resume_scan({"cmd": "--resume"}, deps=deps) is False

    assert allow_initial_resume_scan_error({"allow_initial_resume_scan_error": True}) is True
    assert allow_initial_resume_scan_error({"allow_initial_resume_scan_error": True, "retry_count": 1}) is False
    assert allow_initial_resume_scan_error({"allow_initial_resume_scan_error": True, "resume_from": "/ckpt"}) is False
