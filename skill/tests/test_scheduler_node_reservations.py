from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from skill.scheduler_node.reservations import (
    active_reservations,
    blocked_reserved_nodes,
    make_reservation,
)
from skill.scheduler_submit_command import SubmitCommandDeps, cmd_submit
from skill.scheduler_control_commands import cmd_release_node


@contextmanager
def _lock(*args, **kwargs):
    yield


def _reservation_args(marker):
    return SimpleNamespace(
        signature="campaign/controller", priority="high", description="controller",
        reserve_nodes=["n1"], reserve_signature_prefix="campaign/",
        reserve_until_file=str(marker),
    )


def test_submit_persists_claim_atomically_and_rejects_occupied_node(tmp_path):
    state = {"tasks": []}
    saved = []
    deps = SubmitCommandDeps(
        split_bapr_seed_batch_submit_args=lambda args: [],
        submit_one=lambda args: None,
        run_submit_preflight=lambda args, **kwargs: SimpleNamespace(notes=[]),
        submit_preflight_deps=lambda: None,
        submit_preflight_refusal_type=Exception,
        history_record=lambda *args, **kwargs: None,
        state_lock=_lock,
        load_state=lambda: state,
        build_submit_task_for_state=lambda *args, **kwargs: SimpleNamespace(
            task={"id": "owner", "status": "queued", "submitted_at": 123},
            cpu_cores=1, ram_mb=1000, est_vram=0, source_label="explicit",
        ),
        submit_task_deps=lambda: SimpleNamespace(known_nodes={"n1": {}}),
        save_state=lambda value: saved.append((list(value["tasks"]), list(value["node_reservations"]))),
        submit_task_refusal_type=Exception,
        default_vram_mb=0, default_ram_mb=1000, default_cpu_cores=1,
        print_fn=lambda *args, **kwargs: None,
    )
    args = _reservation_args(tmp_path / "complete.ready")
    cmd_submit(args, deps=deps)
    assert len(saved) == 1
    assert saved[0][0][0]["id"] == "owner"
    assert saved[0][1][0]["owner_task_id"] == "owner"
    assert blocked_reserved_nodes(state, {"id": "bapr", "signature": "BAPR/new"}) == {"n1"}
    assert blocked_reserved_nodes(state, {"id": "wave", "signature": "campaign/wave54"}) == set()

    state["tasks"].append({"id": "bapr", "status": "running", "node": "n1"})
    with pytest.raises(Exception, match="node has active scheduler tasks"):
        make_reservation(
            {"tasks": state["tasks"], "node_reservations": []},
            {"id": "next", "submitted_at": 124}, nodes=["n1"],
            signature_prefix="next/", until_file=str(tmp_path / "next.ready"),
            known_nodes={"n1"},
        )


def test_claim_survives_owner_failure_and_releases_only_after_marker(tmp_path):
    marker = tmp_path / "complete.ready"
    state = {"tasks": [{"id": "owner", "status": "failed"}], "node_reservations": [{
        "owner_task_id": "owner", "nodes": ["n1"],
        "signature_prefix": "campaign/", "until_file": str(marker),
    }]}
    assert active_reservations(state)
    marker.write_text("")
    assert active_reservations(state)
    marker.write_text("terminal and quiescent\n")
    assert active_reservations(state) == []


def test_manual_release_requires_owner_and_wave_to_stop(tmp_path):
    state = {"tasks": [{"id": "owner", "status": "running"}], "node_reservations": [{
        "owner_task_id": "owner", "nodes": ["n1"],
        "signature_prefix": "campaign/", "until_file": str(tmp_path / "marker"),
    }]}
    deps = SimpleNamespace(state_lock=_lock, load_state=lambda: state,
                           save_state=lambda value: None)
    with pytest.raises(SystemExit, match="active campaign tasks"):
        cmd_release_node(SimpleNamespace(owner_task="owner"), deps=deps)
    state["tasks"][0]["status"] = "failed"
    state["tasks"].append({"id": "wave", "status": "running", "node": "n1",
                           "signature": "campaign/wave54"})
    with pytest.raises(SystemExit, match="active campaign tasks"):
        cmd_release_node(SimpleNamespace(owner_task="owner"), deps=deps)
    state["tasks"][1]["status"] = "done"
    state["tasks"].append({"id": "retry", "status": "queued",
                           "signature": "campaign/controller"})
    with pytest.raises(SystemExit, match="active campaign tasks"):
        cmd_release_node(SimpleNamespace(owner_task="owner"), deps=deps)
    state["tasks"][2]["status"] = "cancelled"
    cmd_release_node(SimpleNamespace(owner_task="owner"), deps=deps)
    assert state["node_reservations"] == []
