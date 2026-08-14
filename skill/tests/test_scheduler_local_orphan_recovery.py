from __future__ import annotations

from pathlib import Path

from skill.scheduler_local_orphan_recovery import (
    LocalOrphanRecoveryDeps,
    adopt_live_local_pid_rows,
    recover_queued_live_local_tasks,
    scan_launch_recovery_evidence_on_node,
    scan_scheduleurm_env_pids_on_node,
)
from skill.scheduler_local_launch import runtime_exit_status_path


def _deps(
    tmp_path: Path,
    *,
    run_on=None,
    windows=None,
    requires=None,
    calls=None,
    recover_remote=True,
):
    calls = calls if calls is not None else []

    def set_current_usage(task, vram, ram, pcpu):
        task["current_vram_mb"] = vram
        task["current_ram_mb"] = ram
        task["current_pcpu"] = pcpu

    return LocalOrphanRecoveryDeps(
        state_dir=tmp_path,
        node_configs={
            "node001": {"host": "node001"},
            "node002": {"host": "node002"},
            "win": {"host": "win"},
            "local": {"host": None},
        },
        node_is_windows=windows or (lambda node: node == "win"),
        run_on=run_on or (lambda *args, **kwargs: (1, "", "")),
        remember_last_placement=lambda task: calls.append(("remember", task.get("id"))),
        set_current_usage=set_current_usage,
        claim_enabled_for=lambda node: False,
        update_claim_pid=lambda node, tid, pid: calls.append(("claim", node, tid, pid)),
        requires_local_capacity_check=requires or (lambda node, task=None: True),
        now=lambda: 123.0,
        recover_remote_nodes=recover_remote,
    )


def test_scan_scheduleurm_env_pids_on_node_parses_rows(tmp_path):
    calls = []

    def run_on(node, cmd, timeout=0, check=True):
        calls.append((node, cmd, timeout, check))
        return 0, "t1|11|11|11|S|100|99.5|1234|token-a\nt1|12|11|11|S|50|10\nt2|22|22|22|Z|1|0\nbad\n", ""

    rows = scan_scheduleurm_env_pids_on_node("node001", deps=_deps(tmp_path, run_on=run_on))

    assert calls[0][0] == "node001"
    assert "SCHEDULEURM_TASK_ID" in calls[0][1]
    assert calls[0][2:] == (25, False)
    assert rows == {
        "t1": [
            {
                "pid": "11",
                "sid": "11",
                "pgid": "11",
                "state": "S",
                "rss_mb": "100",
                "pcpu": "99.5",
                "start_ticks": "1234",
                "launch_token": "token-a",
            },
            {"pid": "12", "sid": "11", "pgid": "11", "state": "S", "rss_mb": "50", "pcpu": "10"},
        ],
        "t2": [
            {"pid": "22", "sid": "22", "pgid": "22", "state": "Z", "rss_mb": "1", "pcpu": "0"},
        ],
    }


def test_scan_scheduleurm_env_pids_skips_windows_and_failed_probe(tmp_path):
    assert scan_scheduleurm_env_pids_on_node("win", deps=_deps(tmp_path)) == {}
    assert scan_scheduleurm_env_pids_on_node(
        "node001",
        deps=_deps(tmp_path, run_on=lambda *args, **kwargs: (2, "nope", "")),
    ) == {}


def test_scan_launch_recovery_evidence_batches_processes_and_sentinels(tmp_path):
    calls = []

    def run_on(node, cmd, timeout=0, check=True):
        calls.append((node, cmd, timeout, check))
        if "SCHEDULEURM_TASK_ID" not in cmd:
            return 0, "X\tt2\t0\t456.0\ttoken-b\n", ""
        return 0, "P\tt1\t11\t11\t11\tS\t100\t99.5\t1234\ttoken-a\n", ""

    evidence = scan_launch_recovery_evidence_on_node(
        "node001",
        [
            {"id": "t1", "launch_token": "token-a", "exit_status_path": "/tmp/t1.status"},
            {"id": "t2", "launch_token": "token-b", "exit_status_path": "/tmp/t2 status"},
        ],
        deps=_deps(tmp_path, run_on=run_on),
    )

    assert len(calls) == 2
    assert calls[0][0] == "node001"
    assert calls[0][2:] == (15, False)
    assert "SCHEDULEURM_TASK_ID" not in calls[0][1]
    assert "/tmp/t1.status" in calls[0][1]
    assert "'/tmp/t2 status'" in calls[0][1]
    assert calls[1][2:] == (30, False)
    assert "SCHEDULEURM_TASK_ID" in calls[1][1]
    assert 'os.listdir("/proc")' in calls[1][1]
    assert evidence == {
        "ok": True,
        "supported": True,
        "rows_by_task": {
            "t1": [
                {
                    "pid": "11",
                    "sid": "11",
                    "pgid": "11",
                    "state": "S",
                    "rss_mb": "100",
                    "pcpu": "99.5",
                    "start_ticks": "1234",
                    "launch_token": "token-a",
                }
            ]
        },
        "exit_statuses": {
            "t2": {
                "exit_code": 0,
                "finished_at": 456.0,
                "token": "token-b",
            }
        },
    }


def test_scan_launch_recovery_keeps_sentinels_when_process_scan_fails(tmp_path):
    calls = []

    def run_on(node, cmd, timeout=0, check=True):
        calls.append((node, cmd, timeout, check))
        if "SCHEDULEURM_TASK_ID" not in cmd:
            return 0, "X\tdone\t0\t456.0\tdone-token\n", ""
        raise TimeoutError("process inventory timed out")

    evidence = scan_launch_recovery_evidence_on_node(
        "node001",
        [
            {"id": "done", "launch_token": "done-token"},
            {"id": "unknown", "launch_token": "unknown-token"},
        ],
        deps=_deps(tmp_path, run_on=run_on),
    )

    assert evidence["ok"] is True
    assert evidence["absence_proven"] is False
    assert evidence["exit_statuses"]["done"]["exit_code"] == 0
    assert evidence["rows_by_task"] == {}
    assert "process scan TimeoutError" in evidence["error"]


def test_recover_queued_live_local_tasks_adopts_matching_queued_records(tmp_path):
    calls = []

    def run_on(node, cmd, timeout=0, check=True):
        if node == "node001":
            return 0, "t1|101|101|101|S|120|210.0\nt1|102|101|101|S|80|10\n", ""
        raise AssertionError(f"unexpected scan for {node}")

    state = {
        "tasks": [
            {"id": "t1", "status": "queued", "preferred_node": "node001", "ram_mb": 1, "cpu_cores": 1},
            {"id": "t2", "status": "queued", "preferred_node": "node002"},
            {"id": "t3", "status": "running", "node": "node001"},
            {"id": "t4", "status": "queued", "preferred_node": "win"},
            {"id": "t5", "status": "queued", "preferred_node": "missing"},
        ]
    }

    recovered = recover_queued_live_local_tasks(
        state,
        deps=_deps(
            tmp_path,
            run_on=run_on,
            calls=calls,
            requires=lambda node, task=None: node == "node001",
        ),
    )

    task = state["tasks"][0]
    assert recovered == 1
    assert task["status"] == "running"
    assert task["node"] == "node001"
    assert task["remote_pids"] == [101, 102]
    assert task["process_group"] == 101
    assert task["current_ram_mb"] == 200
    assert task["cpu_cores"] == 3
    assert task["orphan_recovered_from_status"] == "queued"
    assert state["tasks"][1]["status"] == "queued"
    assert calls == [("remember", "t1")]


def test_orphan_recovery_rebuilds_sentinel_and_pid_identity_from_state_token(tmp_path):
    task = {
        "id": "t9",
        "status": "launching",
        "launch_token": "launch-9",
        "launching_started_at": 100.0,
    }
    rows = [
        {
            "pid": "901",
            "sid": "901",
            "pgid": "901",
            "state": "S",
            "rss_mb": "12",
            "pcpu": "1.5",
            "start_ticks": "98765",
        }
    ]

    recovered = adopt_live_local_pid_rows(
        task,
        "node001",
        rows,
        deps=_deps(tmp_path),
        prior_status="launching",
    )

    assert recovered is True
    assert task["status"] == "running"
    assert task["remote_pid_start_ticks"] == {"901": 98765}
    assert task["exit_status_token"] == "launch-9"
    assert task["exit_status_path"] == runtime_exit_status_path("t9", "launch-9")
    assert "launch_token" not in task


def test_orphan_recovery_can_restore_token_from_process_environment(tmp_path):
    task = {"id": "t10", "status": "queued"}
    rows = [
        {
            "pid": "1001",
            "sid": "1001",
            "pgid": "1001",
            "state": "S",
            "start_ticks": "111",
            "launch_token": "remote-token",
        },
        {
            "pid": "1002",
            "sid": "1001",
            "pgid": "1001",
            "state": "S",
            "start_ticks": "222",
            "launch_token": "remote-token",
        },
    ]

    recovered = adopt_live_local_pid_rows(task, "node001", rows, deps=_deps(tmp_path))

    assert recovered is True
    assert task["exit_status_token"] == "remote-token"
    assert task["exit_status_path"] == runtime_exit_status_path("t10", "remote-token")
    assert task["remote_pid_start_ticks"] == {"1001": 111, "1002": 222}


def test_orphan_recovery_rejects_processes_from_another_launch(tmp_path):
    task = {"id": "t11", "status": "launching", "launch_token": "expected-token"}
    rows = [
        {
            "pid": "1101",
            "sid": "1101",
            "pgid": "1101",
            "state": "S",
            "start_ticks": "333",
            "launch_token": "stale-token",
        }
    ]

    recovered = adopt_live_local_pid_rows(task, "node001", rows, deps=_deps(tmp_path))

    assert recovered is False
    assert task["status"] == "launching"
    assert task["launch_token"] == "expected-token"


def test_recover_queued_live_local_tasks_skips_remote_nodes_when_disabled(tmp_path):
    calls = []
    state = {
        "tasks": [
            {"id": "remote", "status": "queued", "preferred_node": "node001"},
            {"id": "localtask", "status": "queued", "preferred_node": "local"},
        ]
    }

    def run_on(node, cmd, timeout=0, check=True):
        calls.append((node, timeout))
        if node == "local":
            return 0, "localtask|201|201|201|S|20|5\n", ""
        raise AssertionError(f"unexpected remote scan for {node}")

    recovered = recover_queued_live_local_tasks(
        state,
        deps=_deps(tmp_path, run_on=run_on, recover_remote=False),
    )

    assert recovered == 1
    assert calls == [("local", 25)]
    assert state["tasks"][0]["status"] == "queued"
    assert state["tasks"][1]["status"] == "running"
