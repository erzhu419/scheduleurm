def test_cpu_slot_accounting_uses_auto_worker_floor(check, sch):
    state = {
        "tasks": [
            {
                "id": "tbatch",
                "status": "running",
                "node": "node006",
                "cpu_cores": 1,
                "cpu_auto_workers": 170,
                "cpu_batch_plan": {"workers": 170},
            },
            {
                "id": "tfreq",
                "status": "running",
                "node": "node006",
                "cpu_cores": 8,
            },
        ]
    }
    nodes = [{"name": "node006", "free_cpu": 168, "total_cpu": 192, "loadavg": 24.0}]
    sch._apply_cpu_slot_accounting_to_nodes(state, nodes)
    check("CPU slot accounting reserves auto-worker floor",
          nodes[0]["cpu_slot_reserved"] == 178,
          diag=f"reserved={nodes[0].get('cpu_slot_reserved')}")
    check("CPU slot accounting free_cpu reflects planned worker slots",
          nodes[0]["free_cpu"] == 14,
          diag=f"free_cpu={nodes[0].get('free_cpu')}")


def test_running_cpu_refresh_preserves_declared_worker_floor(check, sch):
    class FakeBackend:
        def batch_probe(self, state):
            return {
                "tbatch": {
                    "state": "alive",
                    "alive_pids": [1234],
                    "vram_mb": 0,
                    "ram_mb": 100,
                    "pcpu": 50.0,
                }
            }

    task = {
        "id": "tbatch",
        "status": "running",
        "node": "node006",
        "cpu_cores": 170,
        "cpu_auto_workers": 170,
        "cpu_batch_plan": {"workers": 170},
        "cpu_cores_explicit": True,
        "remote_pids": [1234],
        "alive_pids": [1234],
        "started_at": 1,
        "peak_vram_mb": 0,
        "peak_ram_mb": 0,
    }
    state = {"tasks": [task], "next_id": 2}
    old_backend = sch._BACKEND
    try:
        sch._BACKEND = FakeBackend()
        sch._batch_check_running(state)
    finally:
        sch._BACKEND = old_backend
    check("live CPU refresh keeps explicit auto-worker reservation",
          task["cpu_cores"] == 170,
          diag=f"cpu_cores={task.get('cpu_cores')}")


def test_freqduet_soft_pool_prefers_free_cpu_over_generated_pin(check, sch):
    task = {
        "id": "tfreqduet",
        "status": "queued",
        "project": "FreqDuet",
        "signature": "FreqDuet/disc9only_screen_ep100_wu10_r1/shard_0020_0040",
        "description": "FreqDuet config matrix shard",
        "cmd": "python -u scripts/run_freqduet_ablation.py --workers 20",
        "cwd": "/home/erzhu419/mine_code/TransitDuet/FreqDuet/freqduet",
        "est_vram_mb": 0,
        "cpu_cores": 20,
        "ram_mb": 4096,
        "require_node": "node001",
    }
    nodes = [
        {
            "name": "node001",
            "alive": True,
            "gpus": [],
            "free_cpu": 132,
            "total_cpu": 192,
            "free_ram_mb": 132000,
            "total_ram_mb": 192793,
            "running_count": 3,
        },
        {
            "name": "node006",
            "alive": True,
            "gpus": [],
            "free_cpu": 192,
            "total_cpu": 192,
            "free_ram_mb": 179000,
            "total_ram_mb": 192793,
            "running_count": 0,
        },
    ]
    old_runtime = sch._candidate_runtime_seconds
    try:
        sch._candidate_runtime_seconds = lambda _task, node, _gpu_idx=None: 10 if node == "node001" else 0
        placement = sch.pick_placement(task, nodes)
    finally:
        sch._candidate_runtime_seconds = old_runtime
    check("FreqDuet soft CPU pool ignores generated node001 pin and uses free sibling",
          placement == ("node006", None),
          diag=f"placement={placement}")


def test_scheduleurm_cpu_pool_softens_generated_node_pin(check, sch):
    task = {
        "id": "tscheduler",
        "status": "queued",
        "project": "scheduleurm",
        "signature": "scheduleurm/live-oracle/node001-shard",
        "description": "Scheduleurm live oracle CPU task",
        "cmd": "python -u algorithm/experiments/live_scheduler_oracle_closure.py",
        "cwd": "/home/erzhu419/mine_code/scheduleurm",
        "est_vram_mb": 0,
        "cpu_cores": 8,
        "ram_mb": 512,
        "require_node": "node001",
    }
    nodes = [
        {
            "name": "node001",
            "alive": True,
            "gpus": [],
            "free_cpu": 16,
            "total_cpu": 192,
            "free_ram_mb": 100000,
            "total_ram_mb": 192793,
            "running_count": 4,
        },
        {
            "name": "node002",
            "alive": True,
            "gpus": [],
            "free_cpu": 192,
            "total_cpu": 192,
            "free_ram_mb": 160000,
            "total_ram_mb": 192793,
            "running_count": 0,
        },
    ]

    placement = sch.pick_placement(task, nodes)

    check("scheduleurm CPU pool ignores generated node001 pin and uses free sibling",
          placement == ("node002", None),
          diag=f"placement={placement}")


def test_zhengliang_hpc_uses_scheduler_backend(check, sch):
    task = {
        "id": "tbamor",
        "status": "queued",
        "project": "BAMOR",
        "signature": "BAMOR/mujoco/local-scheduler-route",
        "cmd": "python -u train_bamor_mujoco.py",
        "cwd": "/home/erzhu419/mine_code/BAMOR",
        "est_vram_mb": 0,
        "cpu_cores": 8,
        "ram_mb": 8192,
        "require_node": "zhengliang-hpc",
        "slurm_partition": "cpu",
    }
    node_state = {
        "name": "zhengliang-hpc",
        "alive": True,
        "gpus": [],
        "free_cpu": 1216,
        "total_cpu": 1216,
        "free_ram_mb": 1349551,
        "total_ram_mb": 1349551,
        "running_count": 0,
        "slurm_pending_split": {"cpu": 999, "gpu": 999},
    }

    local_check = sch._BACKEND.requires_local_capacity_check(
        "zhengliang-hpc", task, node_state=node_state)
    placement = sch.pick_placement(task, [node_state])

    check("zhengliang-hpc is scheduler-routed, not Slurm-routed",
          local_check,
          diag=f"requires_local_capacity_check={local_check}")
    check("zhengliang-hpc placement ignores Slurm pending throttle when forced local",
          placement == ("zhengliang-hpc", None),
          diag=f"placement={placement}")
