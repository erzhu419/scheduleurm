from __future__ import annotations

from skill.scheduler_adopt.probe import (
    AdoptProbeDeps,
    descendants_of,
    infer_adopt_cwd_from_cmdline,
    node_cpu_processes,
    node_processes,
    refresh_adopted_resources,
)


def _deps(*, run_on=None, windows=False, task_pids=None):
    def default_run_on(node, cmd, **kwargs):
        return 0, "", ""

    return AdoptProbeDeps(
        node_configs={
            "local": {"host": None, "ssh_user": "me", "remote_workspace_root": "/work"},
            "win": {"os": "windows"},
        },
        node_is_windows=lambda node: windows or node == "win",
        run_on=run_on or default_run_on,
        local_user=lambda: "me",
        task_pids=task_pids or (lambda task: list(task.get("remote_pids") or [])),
        descendants_cap=500,
    )


def test_node_processes_parses_gpu_metadata_slurm_and_pipe_cmdline():
    def run_on(node, cmd, **kwargs):
        assert "SLURM_JOB_ID=" in cmd and "SLURM_JOBID=" in cmd
        return (
            0,
            "\n".join([
                "0000:01:00.0, 101, 2048",
                "===BUS===",
                "0, 0000:01:00.0",
                "===META===",
                "101|alice|/work/Proj|4096|125.5|100|1|python train.py --note a|b",
            ]),
            "",
        )

    procs = node_processes("local", deps=_deps(run_on=run_on))

    assert procs == [{
        "node": "local",
        "pid": 101,
        "gpu_idx": 0,
        "used_mb": 2048,
        "owner": "alice",
        "cwd": "/work/Proj",
        "rss_mb": 4,
        "pcpu": 125.5,
        "pgid": 100,
        "cmdline": "python train.py --note a|b",
        "is_slurm": True,
    }]


def test_gpu_adopt_probe_uses_configured_driver_path_for_all_queries():
    from dataclasses import replace

    calls = []
    deps = replace(_deps(run_on=lambda node, cmd, **kwargs: calls.append(cmd) or (0, "", "")),
                   node_configs={"node007": {"nvidia_smi_path": "/driver/bin/nvidia-smi"}})

    assert node_processes("node007", deps=deps) == []
    assert calls[0].count("/driver/bin/nvidia-smi --query-") == 3


def test_node_cpu_processes_parses_cpu_only_processes():
    def run_on(node, cmd, **kwargs):
        assert "${cwd}|${sl}|${cl}" in cmd
        return 0, "202|bob|75.0|2048|200|/work/Eval|0|python eval.py\n", ""

    procs = node_cpu_processes("local", deps=_deps(run_on=run_on))

    assert procs == [{
        "node": "local",
        "pid": 202,
        "owner": "bob",
        "rss_mb": 2,
        "cwd": "/work/Eval",
        "pcpu": 75.0,
        "gpu_idx": None,
        "used_mb": 0,
        "is_cpu_only": True,
        "pgid": 200,
        "cmdline": "python eval.py",
        "is_slurm": False,
    }]


def test_descendants_of_caps_and_infer_cwd_from_cmdline():
    ppid_of = {1000: 1}
    for pid in range(2, 6000):
        ppid_of[pid] = 1000

    out = descendants_of({1000}, ppid_of, cap=500)

    assert len(out) == 500
    assert infer_adopt_cwd_from_cmdline(
        "python /work/ProjectA/train.py --seed 1", ["/work"]
    ) == "/work/ProjectA"


def test_refresh_adopted_resources_updates_cpu_and_downward_ram():
    state = {
        "tasks": [{
            "id": "t1",
            "status": "running",
            "auto_adopted": True,
            "node": "local",
            "remote_pids": [10, 11],
            "cpu_cores": 20,
            "ram_mb": 4096,
        }]
    }
    procs = [
        {"node": "local", "pid": 10, "pcpu": 120.0, "rss_mb": 100, "pgid": 9},
        {"node": "local", "pid": 11, "pcpu": 80.0, "rss_mb": 200, "pgid": 9},
    ]

    refresh_adopted_resources(state, [], [procs], deps=_deps())

    task = state["tasks"][0]
    assert task["cpu_cores"] == 2
    assert task["ram_mb"] == 300
    assert task["process_group"] == 9


def test_cpu_adopt_is_reclassified_when_group_parent_creates_cuda_context():
    task = {
        "id": "t1", "status": "running", "node": "local", "auto_adopted": True,
        "remote_pids": [11], "process_group": 9, "gpu_idx": None,
        "est_vram_mb": 0, "description": "auto-adopted: Train on local:CPU-only",
    }
    state = {"tasks": [task]}
    gpu = {"node": "local", "pid": 10, "pgid": 9, "gpu_idx": 2,
           "used_mb": 4096, "rss_mb": 500, "pcpu": 5}
    cpu = {"node": "local", "pid": 11, "pgid": 9, "rss_mb": 100, "pcpu": 95}

    refresh_adopted_resources(state, [[gpu]], [[cpu]], deps=_deps())

    assert task["gpu_idx"] == 2
    assert task["gpu_indices"] == [2]
    assert task["current_vram_mb"] == task["peak_vram_mb"] == 4096
    assert task["remote_pids"] == [10, 11]
    assert "local:GPU2" in task["description"]
    refresh_adopted_resources(state, [[]], [[cpu]], deps=_deps())
    assert task["gpu_idx"] == 2
