from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from skill.scheduler_commands import edit


@contextmanager
def _noop_lock(**kwargs):
    yield


def _deps(state, *, released=None, saved=None):
    released = released if released is not None else []
    saved = saved if saved is not None else []
    return edit.EditCommandDeps(
        node_configs={"node001": {}, "node002": {}},
        canonicalize_node_list=lambda nodes: list(nodes),
        release_task_claims_and_intents=lambda task, **kwargs: released.append((task["id"], kwargs)),
        state_lock=lambda **kwargs: _noop_lock(**kwargs),
        load_state=lambda: state,
        save_state=lambda state_arg: saved.append(state_arg),
    )


def _args(**overrides):
    base = {
        "id": "t1",
        "project": None,
        "signature": None,
        "confirm": False,
        "vram_mb": None,
        "ram_mb": None,
        "cpu": None,
        "description": None,
        "preferred_node": None,
        "require_node": None,
        "require_gpu_idx": None,
        "allow_gpu_over_one_third": None,
        "allowed_nodes": None,
        "clear_allowed_nodes": False,
        "resource_family": None,
        "vram_resource_family": None,
        "ram_resource_family": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_cmd_edit_updates_resources_and_releases_claims_for_new_preferred_node(capsys):
    released = []
    saved = []
    task = {
        "id": "t1",
        "status": "queued",
        "est_vram_mb": 1000,
        "cpu_cores": 1,
        "preferred_node": "node001",
        "remote_pids": [],
    }
    state = {"tasks": [task]}

    edit.cmd_edit(
        _args(vram_mb=2000, cpu=4, preferred_node="node002", require_gpu_idx="1"),
        deps=_deps(state, released=released, saved=saved),
    )

    assert task["est_vram_mb"] == 2000
    assert task["est_vram_mb_explicit"] is True
    assert task["cpu_cores"] == 4
    assert task["cpu_cores_explicit"] is True
    assert task["preferred_node"] == "node002"
    assert task["require_gpu_idx"] == 1
    assert released == [("t1", {"exclude_nodes": {"node002"}, "clear_markers": False})]
    assert saved == [state]
    out = capsys.readouterr().out
    assert "est_vram_mb: 1000 → 2000" in out
    assert "updated t1" in out


def test_cmd_edit_clears_allowed_nodes_and_gpu_over_one_third():
    released = []
    task = {
        "id": "t1",
        "status": "queued",
        "allowed_nodes": ["node001"],
        "allowed_nodes_user_explicit": True,
        "allowed_nodes_submitted": ["node001"],
        "allowed_nodes_expanded_reason": "unit",
        "allow_gpu_over_one_third": True,
    }
    state = {"tasks": [task]}

    edit.cmd_edit(
        _args(clear_allowed_nodes=True, allow_gpu_over_one_third=False),
        deps=_deps(state, released=released),
    )

    assert "allowed_nodes" not in task
    assert "allowed_nodes_user_explicit" not in task
    assert "allowed_nodes_submitted" not in task
    assert "allowed_nodes_expanded_reason" not in task
    assert "allow_gpu_over_one_third" not in task
    assert released == [("t1", {})]


def test_cmd_edit_rejects_running_tasks():
    state = {"tasks": [{"id": "t1", "status": "running"}]}

    with pytest.raises(SystemExit, match="not queued"):
        edit.cmd_edit(_args(vram_mb=1234), deps=_deps(state))


def test_cmd_edit_moves_generic_resource_family_to_vram_only():
    released = []
    task = {
        "id": "t1",
        "status": "queued",
        "resource_family": "BAPR/new/gpu-runtime",
    }
    state = {"tasks": [task]}

    edit.cmd_edit(
        _args(
            resource_family="",
            vram_resource_family="BAPR/new/gpu-runtime",
        ),
        deps=_deps(state, released=released),
    )

    assert "resource_family" not in task
    assert task["vram_resource_family"] == "BAPR/new/gpu-runtime"
    assert released == [("t1", {})]


def test_cmd_edit_selector_updates_all_matching_queued_tasks_once(capsys):
    released = []
    saved = []
    tasks = [
        {
            "id": "t1",
            "status": "queued",
            "project": "KG-SYNTH",
            "signature": "KG_op/sota/run/a",
            "require_node": "jtl311linux",
            "allowed_nodes": ["jtl311linux"],
        },
        {
            "id": "t2",
            "status": "queued",
            "project": "KG-SYNTH",
            "signature": "KG_op/sota/run/b",
            "require_node": "jtl311linux",
            "allowed_nodes": ["jtl311linux"],
        },
        {
            "id": "t3",
            "status": "running",
            "project": "KG-SYNTH",
            "signature": "KG_op/sota/run/c",
        },
    ]
    state = {"tasks": tasks}
    deps = _deps(state, released=released, saved=saved)
    deps.node_configs.update({
        "jtl110gpu": {},
        "jtl110gpu2": {},
        "node007": {},
        "jtl311linux": {},
    })

    edit.cmd_edit(
        _args(
            id=None,
            project="KG-SYNTH",
            signature="KG_op/sota/run/*",
            confirm=True,
            require_node="",
            allowed_nodes=["jtl110gpu", "jtl110gpu2", "node007"],
        ),
        deps=deps,
    )

    for task in tasks[:2]:
        assert task["require_node"] is None
        assert task["allowed_nodes"] == ["jtl110gpu", "jtl110gpu2", "node007"]
    assert tasks[2].get("allowed_nodes") is None
    assert released == [("t1", {}), ("t2", {})]
    assert saved == [state]
    assert "updated 2 queued task(s)" in capsys.readouterr().out


def test_cmd_edit_selector_requires_confirmation():
    state = {"tasks": [{"id": "t1", "status": "queued", "project": "KG-SYNTH"}]}
    with pytest.raises(SystemExit, match="requires --confirm"):
        edit.cmd_edit(
            _args(id=None, project="KG-SYNTH", vram_mb=1234),
            deps=_deps(state),
        )
