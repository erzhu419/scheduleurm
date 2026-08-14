from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

from skill.scheduler_commands import status


@contextmanager
def _noop_lock(**kwargs):
    yield


def _status_deps(tasks, calls=None):
    calls = calls if calls is not None else []
    return status.StatusCommandDeps(
        status_readonly_lock_timeout_s=0.25,
        lock_timeout_error=TimeoutError,
        running_probe_snapshot_outside_lock=lambda purpose: calls.append(("running_probe", purpose)) or ({}, set()),
        eta_tail_snapshot_outside_lock=lambda purpose: calls.append(("eta_tail", purpose)) or ({}, set()),
        state_lock=lambda **kwargs: calls.append(("lock", kwargs)) or _noop_lock(**kwargs),
        load_state=lambda: {"tasks": tasks},
        recover_stale_launching_tasks=lambda state: calls.append(("recover", None)),
        update_running_tasks=lambda state, **kwargs: calls.append(("update", kwargs)),
        seed_pending_eta_from_history=lambda state: calls.append(("seed_eta", None)),
        reconcile_requeue_lineage_invariants=lambda state: calls.append(("reconcile", None)),
        save_state=lambda state: calls.append(("save", None)),
        compute_node_load_seconds=lambda state: {},
        probe_all=lambda: calls.append(("probe_all", None)) or [],
        apply_cpu_slot_accounting_to_nodes=lambda state, nodes: calls.append(("cpu_slots", None)),
        iter_node_summary_nodes=lambda nodes: nodes,
        node_display_name=lambda node: node,
        format_mem_gb=lambda mb: f"{mb}MB",
        node_configs={},
        format_node_claim_summary=lambda node: "",
        format_node_ram_summary=lambda node: "ram=0/0",
        format_task_location=lambda task: task.get("node") or "-",
        format_task_vram_usage=lambda task: "vram",
        format_task_ram_usage=lambda task: "ram",
        format_task_eta=lambda task: "",
        format_task_owner=lambda task: "owner",
    )


def test_parse_status_filter_ids_accepts_space_and_comma_groups():
    assert status.parse_status_filter_ids(["t1,t2", " t3 ", "", "t2,,t4"]) == {
        "t1",
        "t2",
        "t3",
        "t4",
    }


def test_compact_status_task_keeps_status_surface_and_drops_noise():
    task = {
        "id": "t1",
        "status": "running",
        "description": "desc",
        "node": "node001",
        "remote_pids": [123],
        "slurm_job_id": 99,
        "slurm_state": "FAILED",
        "eta_seconds": 60,
        "last_block_reason": "none",
        "placement_algorithm": "legacy",
        "unexpected_big_payload": "drop",
    }

    compact = status.compact_status_task(task)

    assert compact == {
        "id": "t1",
        "status": "running",
        "description": "desc",
        "node": "node001",
        "remote_pids": [123],
        "slurm_job_id": 99,
        "slurm_state": "FAILED",
        "eta_seconds": 60,
        "last_block_reason": "none",
        "placement_algorithm": "legacy",
    }


def test_filter_status_tasks_keeps_all_when_no_ids_and_matches_stringified_id():
    tasks = [{"id": "t1"}, {"id": 2}, {"id": "t3"}]

    assert status.filter_status_tasks(tasks, set()) == tasks
    assert status.filter_status_tasks(tasks, {"2", "t3"}) == [{"id": 2}, {"id": "t3"}]


def test_status_rows_for_display_hides_terminal_without_all_flag():
    tasks = [
        {"id": "tq", "status": "queued"},
        {"id": "tl", "status": "launching"},
        {"id": "tr", "status": "running"},
        {"id": "td", "status": "done"},
        {"id": "tf", "status": "failed"},
    ]

    assert [t["id"] for t in status.status_rows_for_display(tasks, show_done=False)] == [
        "tq",
        "tl",
        "tr",
    ]
    assert status.status_rows_for_display(tasks, show_done=True) == tasks


def test_status_json_payload_brief_uses_compact_tasks():
    tasks = [{"id": "t1", "status": "done", "unexpected": "drop"}]

    assert status.status_json_payload(tasks, brief=True) == {
        "tasks": [{"id": "t1", "status": "done"}]
    }
    assert status.status_json_payload(tasks, brief=False) == {"tasks": tasks}


def test_format_eta_load_seconds_keeps_legacy_boundaries():
    assert status.format_eta_load_seconds(None) == ""
    assert status.format_eta_load_seconds(0) == ""
    assert status.format_eta_load_seconds(1800) == "  eta_load=30m"
    assert status.format_eta_load_seconds(3600) == "  eta_load=1.0h"
    assert status.format_eta_load_seconds(86399) == "  eta_load=24.0h"
    assert status.format_eta_load_seconds(86400) == "  eta_load=1.0d"


def test_format_task_runtime_minutes_uses_finished_or_current_time():
    assert status.format_task_runtime_minutes({}, now_fn=lambda: 1000.0) == ""
    assert status.format_task_runtime_minutes(
        {"started_at": 100.0, "finished_at": 220.0},
        now_fn=lambda: 1000.0,
    ) == " 2.0m"
    assert status.format_task_runtime_minutes(
        {"started_at": 100.0},
        now_fn=lambda: 250.0,
    ) == " 2.5m"


def test_format_measurement_reservation_exposes_hidden_placement_gate():
    node = {
        "measurement_reservation_active": True,
        "measurement_reservation": {
            "purpose": "theorem-facing natural-completion GPU calibration",
        },
    }

    assert status.format_measurement_reservation(node) == (
        "  MEASUREMENT-RESERVED("
        "theorem-facing natural-completion GPU calibration)"
    )
    assert status.format_measurement_reservation({}) == ""


def test_cmd_status_json_brief_filters_ids(capsys):
    tasks = [
        {"id": "t1", "status": "queued", "description": "one", "unexpected": "drop"},
        {"id": "t2", "status": "done", "description": "two"},
    ]
    args = SimpleNamespace(
        ids=["t1"],
        readonly=True,
        lock_timeout=None,
        json=True,
        brief=True,
        all=False,
    )

    status.cmd_status(args, deps=_status_deps(tasks))

    payload = json.loads(capsys.readouterr().out)
    assert payload == {"tasks": [{"id": "t1", "status": "queued", "description": "one"}]}


def test_cmd_status_readonly_skips_probe_and_mutation(capsys):
    calls = []
    args = SimpleNamespace(
        ids=[],
        readonly=True,
        lock_timeout=None,
        json=False,
        brief=False,
        all=False,
    )

    status.cmd_status(args, deps=_status_deps([], calls=calls))

    out = capsys.readouterr().out
    assert "(readonly: node telemetry/probe skipped)" in out
    assert ("running_probe", "status:running-probe") not in calls
    assert ("probe_all", None) not in calls
    assert not any(call[0] in {"update", "save"} for call in calls)
    lock_calls = [call for call in calls if call[0] == "lock"]
    assert lock_calls and lock_calls[0][1]["shared"] is True
