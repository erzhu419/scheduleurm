from __future__ import annotations

import os

from skill.scheduler_external.reconcile import ExternalReconcileDeps, reconcile_external_tasks


def _proc(
    pid: int,
    *,
    node: str = "local",
    owner: str = "me",
    cwd: str = "/home/me/ProjectA",
    gpu_idx=None,
    used_mb: int = 0,
    rss_mb: int = 128,
    pcpu: float = 95.0,
    pgid: int | None = None,
    cmdline: str = "python train.py",
    is_slurm: bool = False,
):
    return {
        "node": node,
        "pid": pid,
        "owner": owner,
        "cwd": cwd,
        "gpu_idx": gpu_idx,
        "used_mb": used_mb,
        "rss_mb": rss_mb,
        "pcpu": pcpu,
        "pgid": pid if pgid is None else pgid,
        "cmdline": cmdline,
        "is_slurm": is_slurm,
        "is_cpu_only": gpu_idx is None,
    }


def _probe_data(*, gpu=None, cpu=None, ppid=None, owners=None, roots=None, skip_projects=None):
    return {
        "me": "me",
        "adopt_node_names": ["local"],
        "adopt_owner_by_node": {"local": "me"},
        "adopt_owners_by_node": {"local": owners or {"me"}},
        "adopt_roots_by_node": {"local": roots or ["/home/me"]},
        "adopt_skip_projects_by_node": {"local": skip_projects or set()},
        "cleanup_node_names": ["local"],
        "gpu_proc_lists": [gpu or []],
        "cpu_proc_lists": [cpu or []],
        "ppid_maps": [ppid or {}],
    }


def _descendants_of(roots: set, ppid_of: dict) -> set:
    descendants = set()
    changed = True
    while changed:
        changed = False
        for pid, parent in ppid_of.items():
            if pid in descendants:
                continue
            if parent in roots or parent in descendants:
                descendants.add(pid)
                changed = True
    return descendants


def _deps(
    *,
    history=None,
    refresh_calls=None,
    control_predicate=None,
    now_values=None,
):
    history = history or {}
    refresh_calls = refresh_calls if refresh_calls is not None else []
    now_values = iter(now_values or [100.0, 101.0, 102.0, 103.0])

    def allocate_task_id(state):
        next_id = state.setdefault("next_id", 1)
        state["next_id"] = next_id + 1
        return f"t{next_id:04d}"

    def infer_cwd(cmdline, roots):
        for part in cmdline.split():
            if part.startswith(tuple(roots)):
                return os.path.dirname(part) if "." in os.path.basename(part) else part
        return ""

    return ExternalReconcileDeps(
        collect_external_task_probe_data=lambda state: _probe_data(),
        local_user=lambda: "me",
        task_pids=lambda task: list(task.get("remote_pids") or []),
        descendants_of=_descendants_of,
        is_scheduler_control_process=(
            control_predicate
            if control_predicate is not None
            else lambda cwd, cmd: "scheduler.py" in cmd or "pytest" in cmd
        ),
        refresh_adopted_resources=lambda state, gpu, cpu: refresh_calls.append((gpu, cpu)),
        infer_adopt_cwd_from_cmdline=infer_cwd,
        path_under_roots=lambda path, roots: any(
            path == root or path.startswith(root.rstrip("/") + "/") for root in roots
        ),
        project_from_path=lambda path: os.path.basename(path.rstrip("/")),
        history_get=lambda sig: history.get(sig),
        allocate_task_id=allocate_task_id,
        default_ram_mb=4096,
        default_vram_mb=512,
        now=lambda: next(now_values),
    )


def test_adopts_cpu_process_and_uses_history_over_live_estimate():
    refresh_calls = []
    history = {"ProjectA/auto-adopted/p221": {"cpu_cores": 5, "ram_mb": 999}}
    state = {"tasks": [], "next_id": 1}
    proc = _proc(221, rss_mb=0, pcpu=95.0, cmdline="python train.py --seed 1")

    adopted = reconcile_external_tasks(
        state,
        probe_data=_probe_data(cpu=[proc]),
        deps=_deps(history=history, refresh_calls=refresh_calls),
    )

    assert len(adopted) == 1
    task = adopted[0]
    assert task["id"] == "t0001"
    assert task["signature"] == "ProjectA/auto-adopted/p221"
    assert task["est_vram_mb"] == 0
    assert task["cpu_cores"] == 5
    assert task["ram_mb"] == 999
    assert task["cmd"] == "python train.py --seed 1"
    assert task["submitted_at"] == 100.0
    assert task["started_at"] == 101.0
    assert refresh_calls


def test_skips_tracked_descendants_scheduler_pgroups_slurm_and_wrong_owner():
    state = {
        "tasks": [{
            "id": "told",
            "status": "running",
            "node": "local",
            "remote_pids": [10],
            "process_group": 10,
        }],
        "next_id": 1,
    }
    ppid = {11: 10}
    procs = [
        _proc(10),
        _proc(11),
        _proc(12, pgid=10),
        _proc(13, is_slurm=True),
        _proc(14, owner="other"),
        _proc(15, cwd="/home/me/ProjectB"),
    ]

    adopted = reconcile_external_tasks(
        state,
        probe_data=_probe_data(cpu=procs, ppid=ppid, owners={"me"}),
        deps=_deps(),
    )

    assert [task["remote_pids"] for task in adopted] == [[15]]
    assert adopted[0]["project"] == "ProjectB"


def test_forgets_control_process_and_scheduler_child_phantom_adopts():
    state = {
        "tasks": [
            {
                "id": "owned",
                "status": "running",
                "node": "local",
                "remote_pids": [100],
                "process_group": 100,
            },
            {
                "id": "control",
                "status": "running",
                "auto_adopted": True,
                "node": "local",
                "remote_pids": [200],
                "process_group": 200,
                "cwd": "/home/me/scheduleurm",
                "cmd": "python scheduler.py watch",
            },
            {
                "id": "child",
                "status": "running",
                "auto_adopted": True,
                "node": "local",
                "remote_pids": [101],
                "process_group": 101,
                "cwd": "/home/me/ProjectA",
                "cmd": "python worker.py",
            },
            {
                "id": "samepg",
                "status": "running",
                "auto_adopted": True,
                "node": "local",
                "remote_pids": [102],
                "process_group": 100,
                "cwd": "/home/me/ProjectA",
                "cmd": "python worker.py",
            },
        ],
        "next_id": 1,
    }

    adopted = reconcile_external_tasks(
        state,
        probe_data=_probe_data(ppid={101: 100}),
        deps=_deps(now_values=[10.0, 11.0, 12.0]),
    )

    assert adopted == []
    forgotten = {task["id"]: task for task in state["tasks"] if task.get("status") == "forgotten"}
    assert set(forgotten) == {"control", "child", "samepg"}
    assert "scheduler control process" in forgotten["control"]["last_block_reason"]
    assert "duplicate child/process-group" in forgotten["child"]["last_block_reason"]
    assert forgotten["control"]["finished_at"] == 10.0


def test_groups_same_project_by_pgid_and_keeps_worker_group_together():
    state = {"tasks": [], "next_id": 1}
    procs = [
        _proc(301, pgid=300, rss_mb=100, pcpu=40.0),
        _proc(302, pgid=300, rss_mb=200, pcpu=80.0),
        _proc(401, pgid=400, rss_mb=300, pcpu=120.0),
    ]

    adopted = reconcile_external_tasks(
        state,
        probe_data=_probe_data(cpu=procs),
        deps=_deps(now_values=[1.0, 2.0, 3.0, 4.0]),
    )

    assert [task["remote_pids"] for task in adopted] == [[301, 302], [401]]
    assert adopted[0]["process_group"] == 300
    assert adopted[0]["cpu_cores"] == 2
    assert adopted[0]["ram_mb"] == 300
    assert adopted[1]["process_group"] == 400


def test_gpu_process_not_double_counted_when_cpu_probe_reports_same_pid():
    state = {"tasks": [], "next_id": 1}
    proc = _proc(501, gpu_idx=0, used_mb=2048, rss_mb=512, pcpu=60.0)

    adopted = reconcile_external_tasks(
        state,
        probe_data=_probe_data(gpu=[proc], cpu=[dict(proc)]),
        deps=_deps(),
    )

    assert len(adopted) == 1
    assert adopted[0]["remote_pids"] == [501]
    assert adopted[0]["gpu_idx"] == 0
    assert adopted[0]["est_vram_mb"] == 2048
    assert adopted[0]["current_pcpu"] == 60.0
