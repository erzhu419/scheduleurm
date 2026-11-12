import copy
import json
from types import SimpleNamespace

from skill.scheduler_probe.processes import (
    PROCESS_MARKER,
    PROCESS_SCRIPT,
    external_display_tasks,
    parse_process_snapshot,
    process_probe_command,
)


def _proc(pid=101, *, owner="aiyw05", pgid=100, **values):
    return dict(
        pid=pid, ppid=100, pgid=pgid, owner=owner, elapsed_s=600,
        pcpu=90.0, rss_mb=256, comm="fluent", cmdline="fluent -g",
        cwd="", ancestors=[100], gpus=[], **values,
    )


def test_remote_snapshot_includes_other_users_non_python_and_idle_gpu(monkeypatch, capsys):
    import os
    import pwd
    import subprocess
    import sys

    table = "\n".join([
        "aiyw05 101 100 100 600 90.0 262144 fluent fluent -g",
        "aiyw08 102 100 100 600 0.0 4096 python python train.py",
        "researcher 103 100 100 600 0.0 4096 solver solver --gpu",
        "root 104 1 104 600 0.0 2048 sshd sshd -D",
        "aiyw01 105 1 105 600 0.0 1024 bash bash",
        "researcher 106 1 106 600 2.0 262144 Xorg /usr/lib/xorg/Xorg",
        'researcher 107 1 107 1 9.0 4096 python python -c "# scheduleurm-process-snapshot"',
    ])

    def check_output(args, **kwargs):
        if args[0] == "ps":
            return table.encode()
        assert args[0] == "/configured/driver/bin/nvidia-smi"
        if "--query-gpu=uuid,index" in args:
            return b"GPU-a, 2\nGPU-b, 3\n"
        return b"GPU-a, 103, 4096\nGPU-b, 103, 2048\n"

    monkeypatch.setattr(subprocess, "check_output", check_output)
    monkeypatch.setattr(sys, "argv", ["probe", "/configured/driver/bin/nvidia-smi"])
    monkeypatch.setattr(pwd, "getpwnam", lambda user: SimpleNamespace(pw_uid=0 if user == "root" else 1001))
    monkeypatch.setattr(os, "readlink", lambda path: (_ for _ in ()).throw(OSError("permission denied")))
    exec(PROCESS_SCRIPT, {})
    snapshot = json.loads(capsys.readouterr().out)

    assert [proc["pid"] for proc in snapshot["processes"]] == [101, 102, 103]
    assert snapshot["processes"][0]["cwd"] == ""
    assert snapshot["processes"][2]["gpus"] == [{"idx": 2, "used_mb": 4096}, {"idx": 3, "used_mb": 2048}]


def test_process_command_uses_existing_python_without_installing_anything():
    command = process_probe_command("/configured/driver/bin/nvidia-smi")
    assert "command -v python3 || command -v python" in command
    assert PROCESS_MARKER in command
    assert command.endswith(" /configured/driver/bin/nvidia-smi; true")


def test_missing_or_failed_process_snapshot_does_not_break_resource_probe():
    assert parse_process_snapshot("ordinary resource output") == {}
    assert "process_probe_error" in parse_process_snapshot(PROCESS_MARKER + "\n")
    snapshot = parse_process_snapshot(PROCESS_MARKER + '\n{"processes": []}')["process_snapshot"]
    assert snapshot["processes"] == []
    assert snapshot["received_at"] > 0


def test_display_rows_group_workers_without_mutating_queue_or_taking_ownership():
    state = {"tasks": []}
    original = copy.deepcopy(state)
    nodes = [{"name": "node003", "alive": True, "process_snapshot": {
        "sampled_at": 1000, "processes": [_proc(), _proc(102)],
    }}]

    rows = external_display_tasks(state, nodes, is_control_process=lambda *args: False, now=1010)

    assert state == original
    assert len(rows) == 1
    task = rows[0]
    assert task["id"] == "ext-node003-aiyw05-100"
    assert task["process_owner"] == "aiyw05"
    assert task["project"] == "fluent"
    assert task["observed_only"] is True
    assert task["current_ram_mb"] == 512
    assert task["current_pcpu"] == 180.0
    assert task["started_at"] == 400
    assert not task.get("adopted")


def test_display_memory_uses_pss_without_double_counting_forked_workers():
    procs = [_proc(), _proc(102)]
    for proc in procs:
        proc.update(rss_mb=6000, pss_kb=1500 * 1024)
    nodes = [{"name": "node003", "alive": True, "process_snapshot": {"processes": procs}}]
    row, = external_display_tasks({"tasks": []}, nodes, is_control_process=lambda *args: False)
    assert row["current_ram_mb"] == 3000
    assert row["current_ram_measurement"] == "pss"
    del procs[1]["pss_kb"]
    row, = external_display_tasks({"tasks": []}, nodes, is_control_process=lambda *args: False)
    assert row["current_ram_mb"] == 7500
    assert row["current_ram_measurement"] == "rss"


def test_display_rows_deduplicate_tracked_pids_descendants_and_process_groups():
    state = {"tasks": [
        {"id": "t1", "status": "running", "node": "node003", "remote_pids": [101]},
        {"id": "t2", "status": "launching", "node": "node003", "process_group": 200},
        {"id": "t3", "status": "running", "node": "node003", "remote_pid": 401},
    ]}
    procs = [_proc(), _proc(201, pgid=200), _proc(301, pgid=300), _proc(401, pgid=400)]
    procs[2]["ancestors"] = [101]
    procs.append(_proc(501, pgid=500))
    procs[-1]["cmdline"] = "python scheduler.py watch"
    nodes = [{"name": "node003", "process_snapshot": {"processes": procs}}]

    assert external_display_tasks(
        state, nodes, is_control_process=lambda cwd, cmd: "scheduler.py" in cmd,
    ) == []


def test_one_display_row_accounts_for_all_gpus_and_marks_stale_snapshot():
    proc = _proc(owner="researcher")
    proc["gpus"] = [{"idx": 1, "used_mb": 2048}, {"idx": 3, "used_mb": 1024}]
    nodes = [{"name": "node007", "alive": True, "process_snapshot": {
        "sampled_at": 1000, "processes": [proc],
    }}]
    rows = external_display_tasks({"tasks": []}, nodes, is_control_process=lambda *args: False, now=1300)

    assert len(rows) == 1
    assert rows[0]["gpu_indices"] == [1, 3]
    assert rows[0]["current_vram_mb"] == 3072
    assert rows[0]["probe_unknown_since"] == 1000


def test_both_node_probe_paths_preserve_process_snapshots(monkeypatch):
    from skill.scheduler_probe.node import parse_probe_output
    from skill.scheduler_probe.sudo_cpu import parse_cpu_only_sudo_probe

    snapshot = {"sampled_at": 1000, "processes": [_proc()]}
    monkeypatch.setattr("skill.scheduler_probe.processes.time.time", lambda: 1000)
    suffix = "\n" + PROCESS_MARKER + "\n" + json.dumps(snapshot)
    individual = parse_probe_output(
        "node007", {}, "\n===SEP===\n1000\n2000\n===SEP===\n8\n===SEP===\n1\n===SEP===\n===SEP===\n" + suffix,
        deps=None,
    )
    batch = parse_cpu_only_sudo_probe(
        "node003", "1000\n2000\n__SEP__\n8\n__SEP__\n1\n__SEP__\n" + suffix, node_configs={},
    )

    snapshot["received_at"] = 1000
    assert individual["process_snapshot"] == batch["process_snapshot"] == snapshot
    assert individual["free_cpu"] == batch["free_cpu"] == 7


def test_display_elapsed_uses_local_receipt_time_not_skewed_remote_clock():
    nodes = [{"name": "node003", "alive": True, "process_snapshot": {
        "sampled_at": 500, "received_at": 1000, "processes": [_proc()],
    }}]
    task, = external_display_tasks({"tasks": []}, nodes, is_control_process=lambda *args: False, now=1010)
    assert task["started_at"] == 400
    assert "probe_unknown_since" not in task


def test_local_discovery_keeps_real_compute_projects_not_process_or_home_names(tmp_path):
    project = tmp_path / "Hegel Machine"
    project.mkdir()
    (project / "pyproject.toml").touch()
    nested = project / "experiments"
    nested.mkdir()

    def proc(pid, cwd, comm="python3", cmdline="python3 train.py", pcpu=90):
        item = _proc(pid, owner="researcher", pgid=pid)
        item.update(node="local", cwd=str(cwd), comm=comm, cmdline=cmdline, pcpu=pcpu)
        return item

    processes = [
        proc(101, nested),
        proc(102, nested, comm="git", cmdline="git status"),
        proc(103, "", comm="pytho", cmdline="[python3] <defunct>"),
        proc(104, tmp_path, cmdline="python3 result_sync_monitor.py"),
        proc(105, nested, cmdline="python3 -c 'import scheduler'"),
        proc(106, nested, cmdline="python3 continue_batch.py", pcpu=0),
        proc(107, nested, cmdline="python3 -m pytest"),
    ]
    nodes = [{"name": "local", "alive": True, "process_snapshot": {"processes": processes}}]

    rows = external_display_tasks({"tasks": []}, nodes, is_control_process=lambda *args: False)

    assert [(row["project"], row["remote_pids"]) for row in rows] == [("Hegel Machine", [101])]
    assert len(nodes[0]["process_snapshot"]["processes"]) == len(processes)


def test_local_idle_gpu_project_is_kept_without_changing_remote_fallback(tmp_path):
    project = tmp_path / "BAPR"
    project.mkdir()
    (project / ".git").mkdir()
    local = _proc(101, owner="researcher")
    local.update(cwd=str(project), comm="python3", cmdline="python3 train.py", pcpu=0,
                 gpus=[{"idx": 0, "used_mb": 4096}])
    remote = _proc(201)
    nodes = [
        {"name": "local", "alive": True, "process_snapshot": {"processes": [local]}},
        {"name": "node003", "alive": True, "process_snapshot": {"processes": [remote]}},
    ]

    rows = external_display_tasks({"tasks": []}, nodes, is_control_process=lambda *args: False)

    assert [(row["node"], row["project"]) for row in rows] == [("local", "BAPR"), ("node003", "fluent")]
