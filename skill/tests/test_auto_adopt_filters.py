import json


def _probe_data_for_processes(processes, *, owner="erzhu419", roots=None):
    roots = roots or ["/home/erzhu419"]
    return {
        "me": owner,
        "adopt_node_names": ["local"],
        "adopt_owner_by_node": {"local": owner},
        "adopt_owners_by_node": {"local": {owner}},
        "adopt_roots_by_node": {"local": roots},
        "adopt_skip_projects_by_node": {"local": set()},
        "cleanup_node_names": ["local"],
        "gpu_proc_lists": [[]],
        "cpu_proc_lists": [processes],
        "ppid_maps": [{}],
    }


def _cpu_proc(pid, *, owner="erzhu419", cwd="/home/erzhu419/mine_code/scheduleurm", cmdline="python3 -m pytest -q"):
    return {
        "node": "local",
        "pid": pid,
        "owner": owner,
        "rss_mb": 128,
        "cwd": cwd,
        "pcpu": 95.0,
        "gpu_idx": None,
        "used_mb": 0,
        "is_cpu_only": True,
        "pgid": pid,
        "cmdline": cmdline,
        "is_slurm": False,
    }


def test_auto_adopt_skips_scheduleurm_pytest_and_stdin_helpers(sch):
    state = {"tasks": [], "next_id": 1}
    processes = [
        _cpu_proc(111, cmdline="python3 -m pytest -q"),
        _cpu_proc(112, cmdline="python3 -"),
        _cpu_proc(113, cmdline="python3 -u -"),
        _cpu_proc(114, cmdline="python3 skill/scheduler.py show watcher-state"),
        _cpu_proc(115, cmdline="python3 ./skill/scheduler.py status --readonly"),
        _cpu_proc(116, cmdline="python3 skill/test_regression.py"),
    ]

    adopted = sch._reconcile_external_tasks(
        state,
        probe_data=_probe_data_for_processes(
            processes,
            roots=["/home/erzhu419"],
        ),
    )

    assert adopted == []
    assert state["tasks"] == []


def test_scheduler_control_cmdline_detects_relative_scheduler_path(sch):
    assert sch._is_scheduler_control_cmdline("python3 skill/scheduler.py status --json")
    assert sch._is_scheduler_control_cmdline("python3 ./skill/scheduler.py show t1")


def test_auto_adopt_keeps_real_external_workloads(sch):
    owner = "zhengliang01"
    cwd = "/home/zhengliang01/scheduleurm_work/TransitDuet"
    state = {"tasks": [], "next_id": 1}
    processes = [
        _cpu_proc(
            221,
            owner=owner,
            cwd=cwd,
            cmdline="python train_transit.py --seed 1",
        )
    ]

    adopted = sch._reconcile_external_tasks(
        state,
        probe_data=_probe_data_for_processes(
            processes,
            owner=owner,
            roots=["/home/zhengliang01"],
        ),
    )

    assert len(adopted) == 1
    assert adopted[0]["project"] == "TransitDuet"
    assert adopted[0]["status"] == "running"
    assert adopted[0]["signature"] == "TransitDuet/auto-adopted/p221"


def test_auto_forgets_existing_running_scheduleurm_control_adopt(sch):
    task = {
        "id": "told",
        "status": "running",
        "auto_adopted": True,
        "node": "local",
        "remote_pids": [331],
        "process_group": 331,
        "cwd": str(sch._SCHEDULEURM_ROOT),
        "cmd": "python3 skill/test_regression.py",
    }
    state = {"tasks": [task], "next_id": 2}
    process = _cpu_proc(
        331,
        cwd=str(sch._SCHEDULEURM_ROOT),
        cmdline="python3 skill/test_regression.py",
    )

    adopted = sch._reconcile_external_tasks(
        state,
        probe_data=_probe_data_for_processes(
            [process],
            roots=["/home/erzhu419"],
        ),
    )

    assert adopted == []
    assert task["status"] == "forgotten"
    assert "scheduler control process" in task["last_block_reason"]


def test_archive_terminal_control_plane_autoadopts_immediately(sch):
    now = sch.time.time()
    control = {
        "id": "tcontrol",
        "status": "done",
        "auto_adopted": True,
        "signature": "scheduleurm/auto-adopted/p123",
        "cwd": str(sch._SCHEDULEURM_ROOT),
        "cmd": "python3 skill/test_regression.py",
        "finished_at": now,
    }
    real_external = {
        "id": "treal",
        "status": "done",
        "auto_adopted": True,
        "signature": "TransitDuet/auto-adopted/p456",
        "cwd": "/home/zhengliang01/scheduleurm_work/TransitDuet",
        "cmd": "python train_transit.py --seed 1",
        "finished_at": now,
    }
    running_control = {
        "id": "trunning",
        "status": "running",
        "auto_adopted": True,
        "signature": "scheduleurm/auto-adopted/p789",
        "cwd": str(sch._SCHEDULEURM_ROOT),
        "cmd": "python3 -m pytest -q",
    }
    state = {"tasks": [control, real_external, running_control], "next_id": 4}

    archived = sch.archive_terminal_tasks(state, age_days=999, max_hot_terminal=500)

    assert archived == 1
    assert [task["id"] for task in state["tasks"]] == ["treal", "trunning"]
    archived_rows = [json.loads(line) for line in sch.ARCHIVE_FILE.read_text().splitlines()]
    assert [task["id"] for task in archived_rows] == ["tcontrol"]


def test_auto_adopt_probe_reuses_alive_node_filter(sch, monkeypatch):
    saved_nodes = dict(sch.NODES)
    calls = []
    try:
        sch.NODES.clear()
        sch.NODES.update({
            "down": {"host": "down.example"},
            "alive": {"host": "alive.example"},
        })
        monkeypatch.setattr(
            sch,
            "_node_processes",
            lambda node: calls.append(("gpu", node)) or [],
        )
        monkeypatch.setattr(
            sch,
            "_node_cpu_processes",
            lambda node: calls.append(("cpu", node)) or [],
        )
        monkeypatch.setattr(
            sch,
            "_node_ppid_map",
            lambda node: calls.append(("ppid", node)) or {},
        )

        data = sch._collect_external_task_probe_data(
            eligible_node_names={"alive"}
        )

        assert data["adopt_node_names"] == ["alive"]
        assert data["cleanup_node_names"] == ["alive"]
        assert calls == [
            ("gpu", "alive"),
            ("cpu", "alive"),
            ("ppid", "alive"),
        ]
    finally:
        sch.NODES.clear()
        sch.NODES.update(saved_nodes)
