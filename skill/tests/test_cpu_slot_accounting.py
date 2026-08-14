from skill.scheduler_cpu.capacity import apply_cpu_slot_accounting_to_nodes


def test_hpc_startup_hold_decays_toward_observed_usage():
    state = {
        "tasks": [{
            "id": "t1",
            "status": "running",
            "node": "node001",
            "cpu_cores": 12,
            "started_at": 955.0,
            "current_pcpu": 0.0,
        }]
    }
    nodes = [{"name": "node001", "free_cpu": 48, "total_cpu": 192}]
    configs = {
        "node001": {
            "cpu_cores": 192,
            "live_cpu_backfill_startup_grace_s": 90,
        }
    }

    apply_cpu_slot_accounting_to_nodes(
        state,
        nodes,
        node_configs=configs,
        cpu_slot_accounting_node_names=["node001"],
        reserved_cpu_slots_for_task=lambda task: int(task["cpu_cores"]),
        now_fn=lambda: 1000.0,
    )

    assert nodes[0]["cpu_hard_reserved"] == 6
    assert nodes[0]["cpu_hard_free"] == 42
    assert nodes[0]["cpu_startup_hold_remaining_s"] == 45


def test_single_thread_startup_hold_uses_short_grace():
    state = {
        "tasks": [{
            "id": "single",
            "status": "running",
            "node": "node001",
            "cpu_cores": 1,
            "started_at": 995.0,
            "current_pcpu": 0.0,
        }]
    }
    nodes = [{"name": "node001", "free_cpu": 48, "total_cpu": 192}]
    configs = {
        "node001": {
            "cpu_cores": 192,
            "live_cpu_backfill_startup_grace_s": 90,
            "live_cpu_backfill_single_thread_startup_grace_s": 10,
        }
    }

    apply_cpu_slot_accounting_to_nodes(
        state,
        nodes,
        node_configs=configs,
        cpu_slot_accounting_node_names=["node001"],
        reserved_cpu_slots_for_task=lambda task: int(task["cpu_cores"]),
        now_fn=lambda: 1000.0,
    )

    assert nodes[0]["cpu_hard_reserved"] == 1
    assert nodes[0]["cpu_hard_free"] == 47
    assert nodes[0]["cpu_startup_hold_remaining_s"] == 5


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


def test_hpc_starting_wide_tasks_keep_hard_reservations_with_live_probe(check, sch):
    state = {
        "tasks": [
            {"id": f"t{i}", "status": "running", "node": "node006", "cpu_cores": 40}
            for i in range(4)
        ]
    }
    nodes = [{
        "name": "node006",
        "free_cpu": 96,
        "total_cpu": 192,
        "loadavg": 96.0,
        "free_ram_mb": 170000,
        "total_ram_mb": 192793,
        "running_count": 4,
    }]
    sch._apply_cpu_slot_accounting_to_nodes(state, nodes)
    task = {
        "id": "tqueued",
        "status": "queued",
        "project": "FreqDuet",
        "est_vram_mb": 0,
        "cpu_cores": 40,
        "ram_mb": 9053,
    }

    ok, why = sch._node_resources_ok(task, nodes[0], sch.NODES["node006"])

    check("slot accounting still records conservative reservation free",
          nodes[0]["free_cpu"] == 32,
          diag=f"free_cpu={nodes[0].get('free_cpu')}")
    check("starting wide HPC tasks reserve their unobserved startup load",
          (not ok) and "cpu: need 40, free 0/192" in why,
          diag=f"ok={ok} why={why} node={nodes[0]}")


def test_hpc_mature_single_core_task_can_use_guarded_live_backfill(check, sch):
    state = {
        "tasks": [
            {
                "id": f"t{i}",
                "status": "running",
                "node": "node006",
                "cpu_cores": 1,
                "started_at": 1,
            }
            for i in range(192)
        ]
    }
    nodes = [{
        "name": "node006",
        "free_cpu": 48,
        "total_cpu": 192,
        "loadavg": 144.0,
        "free_ram_mb": 170000,
        "total_ram_mb": 192793,
        "running_count": 192,
    }]
    sch._apply_cpu_slot_accounting_to_nodes(state, nodes)
    task = {
        "id": "tqueued",
        "status": "queued",
        "project": "single-core-backfill",
        "est_vram_mb": 0,
        "cpu_cores": 1,
        "ram_mb": 1024,
    }

    free, source, slot_free, observed_free = sch._node_schedulable_cpu_free(
        task, nodes[0], sch.NODES["node006"]
    )

    check("mature single-core jobs remain eligible for guarded live backfill",
          (free, source, slot_free, observed_free) == (28, "live_guarded", 0, 48),
          diag=f"free={free} source={source} node={nodes[0]}")


def test_hpc_launching_single_core_wave_blocks_stale_probe_backfill(check, sch):
    state = {
        "tasks": [
            {
                "id": f"t{i}",
                "status": "launching",
                "node": "node006",
                "cpu_cores": 1,
                "launching_started_at": 1,
            }
            for i in range(192)
        ]
    }
    nodes = [{
        "name": "node006",
        "free_cpu": 192,
        "total_cpu": 192,
        "loadavg": 0.0,
        "free_ram_mb": 170000,
        "total_ram_mb": 192793,
        "running_count": 192,
    }]
    sch._apply_cpu_slot_accounting_to_nodes(state, nodes)
    task = {
        "id": "tqueued",
        "status": "queued",
        "project": "single-core-backfill",
        "est_vram_mb": 0,
        "cpu_cores": 1,
        "ram_mb": 1024,
    }

    free, source, slot_free, observed_free = sch._node_schedulable_cpu_free(
        task, nodes[0], sch.NODES["node006"]
    )

    check("launching one-core wave reserves capacity before CPU utilization rises",
          (free, source, slot_free, observed_free) == (0, "live_guarded", 0, 192),
          diag=f"free={free} source={source} node={nodes[0]}")


def test_hpc_cpu_slot_accounting_backfill_still_rejects_live_cpu_shortage(check, sch):
    state = {
        "tasks": [
            {
                "id": f"t{i}",
                "status": "running",
                "node": "node006",
                "cpu_cores": 40,
                "started_at": 1,
            }
            for i in range(4)
        ]
    }
    nodes = [{
        "name": "node006",
        "free_cpu": 20,
        "total_cpu": 192,
        "loadavg": 172.0,
        "free_ram_mb": 170000,
        "total_ram_mb": 192793,
        "running_count": 4,
    }]
    sch._apply_cpu_slot_accounting_to_nodes(state, nodes)
    task = {
        "id": "tqueued",
        "status": "queued",
        "project": "FreqDuet",
        "est_vram_mb": 0,
        "cpu_cores": 40,
        "ram_mb": 9053,
    }

    ok, why = sch._node_resources_ok(task, nodes[0], sch.NODES["node006"])

    check("wide HPC CPU task obeys the tighter live CPU limit",
          (not ok) and "cpu: need 40, free 0/192" in why,
          diag=f"ok={ok} why={why} node={nodes[0]}")


def test_hpc_wide_cpu_task_rejects_zero_live_free_despite_open_slots(check, sch):
    state = {
        "tasks": [
            {
                "id": f"t{i}",
                "status": "running",
                "node": "node004",
                "cpu_cores": 12,
                "started_at": 1,
            }
            for i in range(15)
        ]
    }
    nodes = [{
        "name": "node004",
        "free_cpu": 0,
        "total_cpu": 192,
        "loadavg": 192.0,
        "free_ram_mb": 170000,
        "total_ram_mb": 192793,
        "running_count": 15,
    }]
    sch._apply_cpu_slot_accounting_to_nodes(state, nodes)
    task = {
        "id": "tqueued",
        "status": "queued",
        "project": "KG-SYNTH",
        "est_vram_mb": 0,
        "cpu_cores": 12,
        "ram_mb": 4096,
    }

    free, source, slot_free, observed_free = sch._node_schedulable_cpu_free(
        task, nodes[0], sch.NODES["node004"]
    )
    ok, why = sch._node_resources_ok(task, nodes[0], sch.NODES["node004"])

    check("live saturation overrides apparently open declared slots",
          (free, source, slot_free, observed_free)
          == (0, "live_guarded", 12, 0),
          diag=f"free={free} source={source} node={nodes[0]}")
    check("wide task is rejected on a live-saturated node",
          (not ok) and "cpu: need 12, free 0/192" in why,
          diag=f"ok={ok} why={why}")


def test_hpc_mature_wide_tasks_can_backfill_real_idle_cpu(check, sch):
    state = {
        "tasks": [
            {
                "id": f"t{i}",
                "status": "running",
                "node": "node002",
                "cpu_cores": 12,
                "started_at": 1,
            }
            for i in range(15)
        ]
    }
    nodes = [{
        "name": "node002",
        "free_cpu": 47,
        "total_cpu": 192,
        "loadavg": 145.0,
        "free_ram_mb": 170000,
        "total_ram_mb": 192793,
        "running_count": 15,
    }]
    sch._apply_cpu_slot_accounting_to_nodes(state, nodes)
    task = {
        "id": "tqueued",
        "status": "queued",
        "project": "KG-SYNTH",
        "est_vram_mb": 0,
        "cpu_cores": 12,
        "ram_mb": 4096,
    }

    free, source, slot_free, observed_free = sch._node_schedulable_cpu_free(
        task, nodes[0], sch.NODES["node002"]
    )
    ok, why = sch._node_resources_ok(task, nodes[0], sch.NODES["node002"])

    check("mature wide jobs use real idle CPU instead of permanent slot floors",
          (free, source, slot_free, observed_free)
          == (27, "live_guarded", 12, 47),
          diag=f"free={free} source={source} node={nodes[0]}")
    check("12-core task fits in observed 47-core headroom",
          ok and why == "ok",
          diag=f"ok={ok} why={why}")


def test_hpc_startup_reservation_only_covers_unobserved_cpu(check, sch):
    state = {
        "tasks": [
            {
                "id": f"t{i}",
                "status": "running",
                "node": "node003",
                "cpu_cores": 12,
                "started_at": 10**20,
                "current_pcpu": 640.0,
            }
            for i in range(5)
        ]
    }
    nodes = [{
        "name": "node003",
        "free_cpu": 65,
        "total_cpu": 192,
        "loadavg": 127.0,
        "free_ram_mb": 170000,
        "total_ram_mb": 192793,
        "running_count": 5,
    }]
    sch._apply_cpu_slot_accounting_to_nodes(state, nodes)
    task = {
        "id": "tqueued",
        "status": "queued",
        "project": "KG-SYNTH",
        "est_vram_mb": 0,
        "cpu_cores": 12,
        "ram_mb": 4096,
    }

    free, source, slot_free, observed_free = sch._node_schedulable_cpu_free(
        task, nodes[0], sch.NODES["node003"]
    )

    check("observed startup CPU is not reserved a second time",
          nodes[0]["cpu_hard_reserved"] == 30,
          diag=f"node={nodes[0]}")
    check("remaining startup deficit leaves schedulable headroom",
          (free, source, slot_free, observed_free)
          == (15, "live_guarded", 132, 65),
          diag=f"free={free} source={source} node={nodes[0]}")


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


def test_freqduet_external_baseline_soft_pool_uses_free_sibling(check, sch):
    task = {
        "id": "tfreqduet-external",
        "status": "queued",
        "project": "FreqDuet",
        "signature": "FreqDuet/paper_route_day_fixed_v1b_ep100_20seed/shard_0800_0840",
        "description": "FreqDuet external baselines shard",
        "cmd": (
            "PYTHONPATH=. python -u scripts/run_freqduet_external_baselines.py "
            "--workers 40 --worker-threads 1"
        ),
        "cwd": "/home/erzhu419/mine_code/TransitDuet/FreqDuet/freqduet",
        "est_vram_mb": 0,
        "cpu_cores": 40,
        "ram_mb": 9053,
        "require_node": "node003",
    }
    nodes = [
        {
            "name": "node003",
            "alive": True,
            "gpus": [],
            "free_cpu": 12,
            "total_cpu": 192,
            "free_ram_mb": 132000,
            "total_ram_mb": 192793,
            "running_count": 4,
        },
        {
            "name": "node006",
            "alive": True,
            "gpus": [],
            "free_cpu": 136,
            "total_cpu": 192,
            "free_ram_mb": 179000,
            "total_ram_mb": 192793,
            "running_count": 2,
        },
    ]

    placement = sch.pick_placement(task, nodes)

    check("FreqDuet external baseline node pin is softened to the CPU pool",
          placement == ("node006", None),
          diag=f"placement={placement}")


def test_transitduet_runner_soft_pool_uses_free_sibling(check, sch):
    task = {
        "id": "ttransitduet",
        "status": "queued",
        "project": "TransitDuet",
        "signature": "TransitDuet/H_hiro/node001-seed42",
        "description": "TransitDuet train H_hiro seed 42",
        "cmd": (
            "/home/zhengliang01/scheduleurm_work/conda_envs/"
            "freqduet-cpu-py310/bin/python -u runner_v3.py "
            "--config configs_ablation/H_hiro.yaml --episodes 300 --seed 42"
        ),
        "cwd": "/home/erzhu419/mine_code/TransitDuet/transit_duet",
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
            "free_cpu": 8,
            "total_cpu": 192,
            "free_ram_mb": 132000,
            "total_ram_mb": 192793,
            "running_count": 23,
        },
        {
            "name": "node006",
            "alive": True,
            "gpus": [],
            "free_cpu": 160,
            "total_cpu": 192,
            "free_ram_mb": 179000,
            "total_ram_mb": 192793,
            "running_count": 4,
        },
    ]

    placement = sch.pick_placement(task, nodes)

    check("TransitDuet runner_v3 node pin is softened to the CPU pool",
          placement == ("node006", None),
          diag=f"placement={placement}")


def test_live_orphan_recovery_tracks_all_node_worker_children(check, sch):
    task = {
        "id": "tworker",
        "status": "queued",
        "project": "TransitDuet",
        "signature": "TransitDuet/node-worker/node003/seeds2003-2060/v1",
        "description": "TransitDuet node worker node003 seeds 2003-2060",
        "cmd": "python -u scripts/run_round3_node_worker.py --max-procs 140",
        "cwd": "/home/erzhu419/mine_code/TransitDuet/transit_duet",
        "require_node": "node003",
        "cpu_cores": 117,
        "ram_mb": 704,
        "est_vram_mb": 0,
    }
    rows = [
        {"pid": "629460", "sid": "629460", "pgid": "629460", "state": "R", "rss_mb": "5402", "pcpu": "92.5"},
        {"pid": "629461", "sid": "629461", "pgid": "629461", "state": "R", "rss_mb": "747", "pcpu": "92.3"},
        {"pid": "629463", "sid": "629463", "pgid": "629463", "state": "R", "rss_mb": "710", "pcpu": "92.5"},
    ]

    ok = sch._adopt_live_local_pid_rows(task, "node003", rows, prior_status="queued")

    check("queued live orphan recovery succeeds",
          ok and task["status"] == "running" and task["node"] == "node003",
          diag=f"task={task}")
    check("node-worker recovery keeps all child PIDs",
          task["remote_pids"] == [629460, 629461, 629463],
          diag=f"remote_pids={task.get('remote_pids')}")
    check("node-worker recovery sums RAM and bumps estimate",
          task["current_ram_mb"] == 6859 and task["ram_mb"] == 6859,
          diag=f"current_ram={task.get('current_ram_mb')} ram={task.get('ram_mb')}")
    check("independent child process groups are not collapsed",
          task.get("process_group") is None,
          diag=f"process_group={task.get('process_group')}")


def test_transitduet_baseline_and_node_worker_soft_pool_uses_free_sibling(check, sch):
    nodes = [
        {
            "name": "node001",
            "alive": True,
            "gpus": [],
            "free_cpu": 66,
            "total_cpu": 192,
            "free_ram_mb": 7908,
            "total_ram_mb": 192793,
            "running_count": 10,
            "observed_free_cpu": 66,
            "cpu_slot_accounting": True,
        },
        {
            "name": "node003",
            "alive": True,
            "gpus": [],
            "free_cpu": 143,
            "total_cpu": 192,
            "free_ram_mb": 170947,
            "total_ram_mb": 192793,
            "running_count": 1,
            "observed_free_cpu": 143,
            "cpu_slot_accounting": True,
        },
    ]
    baseline = {
        "id": "tbaseline",
        "status": "queued",
        "project": "TransitDuet",
        "signature": "TransitDuet/baseline/ga/seed1006/episodes300",
        "description": "TransitDuet baseline ga seed 1006",
        "cmd": (
            "/home/zhengliang01/scheduleurm_work/conda_envs/"
            "freqduet-cpu-py310/bin/python -u run_upper_comparison.py "
            "--method ga --episodes 300 --lower_warmup 20 --seed 1006"
        ),
        "cwd": "/home/erzhu419/mine_code/TransitDuet/transit_duet",
        "est_vram_mb": 0,
        "cpu_cores": 1,
        "ram_mb": 902,
        "require_node": "node001",
    }
    node_worker = {
        "id": "tnodeworker",
        "status": "queued",
        "project": "TransitDuet",
        "signature": "TransitDuet/node-worker/node001/seeds2003-2060/v1",
        "description": "TransitDuet node worker node001 seeds 2003-2060",
        "cmd": (
            "/home/zhengliang01/scheduleurm_work/conda_envs/"
            "freqduet-cpu-py310/bin/python -u scripts/run_round3_node_worker.py "
            "--node-index 0 --node-count 6 --seeds 2003:2060 --episodes 300 "
            "--max-procs 170 --root /home/zhengliang01/scheduleurm_work/"
            "TransitDuet/transit_duet"
        ),
        "cwd": "/home/erzhu419/mine_code/TransitDuet/transit_duet",
        "est_vram_mb": 0,
        "cpu_cores": 118,
        "ram_mb": 704,
        "require_node": "node001",
    }

    baseline_placement = sch.pick_placement(baseline, nodes)
    worker_placement = sch.pick_placement(node_worker, nodes)

    check("TransitDuet baseline node pin is softened to the CPU pool",
          baseline_placement == ("node003", None),
          diag=f"placement={baseline_placement}")
    check("TransitDuet node-worker node pin is softened to the CPU pool",
          worker_placement == ("node003", None),
          diag=f"placement={worker_placement}")


def test_transitduet_done_results_marker_is_success(check, sch, tmp_path):
    log_path = tmp_path / "transitduet_done.log"
    log_path.write_text(
        "TransitDuet v3 [timetable] | eps=300 | dev=cpu\n"
        "  [Checkpoint ep 299]\n\n"
        "Done. Results in /home/zhengliang01/scheduleurm_work/TransitDuet/transit_duet/logs/H_fixed_timetable_seed42/\n",
        encoding="utf-8",
    )
    task = {
        "id": "tdone",
        "status": "running",
        "project": "TransitDuet",
        "node": "local",
        "log_path": str(log_path),
        "started_at": 1000.0,
        "finished_at": 1300.0,
        "cmd": "python -u runner_v3.py --episodes 300",
    }

    diag = sch._diagnose_terminal(task)

    check("TransitDuet Done. Results in marker is not classified as crash",
          not diag.get("is_crash"),
          diag=f"diag={diag}")
    check("TransitDuet success marker is recorded",
          diag.get("success_marker") == "Done. Results in",
          diag=f"diag={diag}")


def test_transitduet_baseline_and_rule_final_lines_are_success(check, sch, tmp_path):
    cases = [
        ("baseline", "Results saved to /home/zhengliang01/scheduleurm_work/TransitDuet/transit_duet/logs/upper_ga_seed123\n",
         "Results saved to"),
        ("rule", "ep299  N=11  H=(360.0, 360.0, 360.0)  wait=5.51\nBest composite=0.741 at ep212 with H=[360.0, 360.0, 360.0]\n",
         "Best composite="),
        ("headway", "Best headway: H_peak=247s, H_off=350s, H_trans=320s\n",
         "Best headway:"),
    ]
    for name, tail, marker in cases:
        log_path = tmp_path / f"{name}.log"
        log_path.write_text(
            "[Ep 295] TransitDuet final evaluation\n" + tail,
            encoding="utf-8",
        )
        task = {
            "id": f"t-{name}",
            "status": "running",
            "project": "TransitDuet",
            "node": "local",
            "log_path": str(log_path),
            "started_at": 1000.0,
            "finished_at": 6500.0,
            "cmd": "python -u run_upper_comparison.py --episodes 300",
        }

        diag = sch._diagnose_terminal(task)

        check(f"TransitDuet {marker} marker is not classified as crash",
              not diag.get("is_crash"),
              diag=f"name={name} diag={diag}")
        check(f"TransitDuet {marker} marker is recorded",
              diag.get("success_marker") == marker,
              diag=f"name={name} diag={diag}")


def test_transitduet_node_worker_complete_zero_failures_is_success(check, sch, tmp_path):
    log_path = tmp_path / "node_worker.log"
    log_path.write_text(
        "node-worker index=4/6 tasks=154 max_procs=140 root=/work/TransitDuet/transit_duet\n"
        "node-worker complete failures=0\n",
        encoding="utf-8",
    )
    task = {
        "id": "tnodeworker-done",
        "status": "running",
        "project": "TransitDuet",
        "node": "local",
        "log_path": str(log_path),
        "started_at": 1000.0,
        "finished_at": 8200.0,
        "cmd": "python -u scripts/run_round3_node_worker.py --max-procs 140",
    }

    diag = sch._diagnose_terminal(task)

    check("TransitDuet node-worker failures=0 marker is not classified as crash",
          not diag.get("is_crash"),
          diag=f"diag={diag}")
    check("TransitDuet node-worker success marker is recorded",
          diag.get("success_marker") == "node-worker complete failures=0",
          diag=f"diag={diag}")


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


def test_zhengliang_hpc_cpu_tasks_route_to_worker_pool(check, sch):
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
    login_state = {
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
    worker_state = {
        "name": "node006",
        "alive": True,
        "gpus": [],
        "free_cpu": 176,
        "total_cpu": 192,
        "free_ram_mb": 178000,
        "total_ram_mb": 192793,
        "running_count": 0,
    }

    local_check = sch._BACKEND.requires_local_capacity_check(
        "zhengliang-hpc", task, node_state=login_state)
    placement = sch.pick_placement(task, [login_state, worker_state])
    login_explain = sch._explain_node_fit(task, login_state)

    check("zhengliang-hpc backend remains scheduler-routed, not Slurm-routed",
          local_check,
          diag=f"requires_local_capacity_check={local_check}")
    check("zhengliang-hpc CPU training tag routes to node001-node006 workers",
          placement == ("node006", None),
          diag=f"placement={placement}")
    check("zhengliang-hpc is not an execution candidate for the CPU pool",
          "login-node-disabled" in login_explain,
          diag=login_explain)


def test_probe_all_collapses_shared_outer_route_failure(check, sch, monkeypatch):
    monkeypatch.setattr(sch, "NODES", {
        "local": {"host": None, "cpu_cores": 1, "ram_mb": 1024},
        "node001": {
            "host": "202.197.46.16",
            "ssh_user": "zhengliang01",
            "ssh_proxy_jumps": ["jtl110gpu2", "jtl110gpu"],
            "sudo_ssh_host": "node001",
            "cpu_cores": 192,
        },
        "node002": {
            "host": "202.197.46.16",
            "ssh_user": "zhengliang01",
            "ssh_proxy_jumps": ["jtl110gpu2", "jtl110gpu"],
            "sudo_ssh_host": "node002",
            "cpu_cores": 192,
        },
    })
    monkeypatch.setattr(sch, "_probe_ssh_proxy_jump", lambda name, jump, timeout=8: False)
    monkeypatch.setattr(sch, "_probe_outer_ssh_route",
                        lambda name: (False, "outer ssh route timed out after 3s"))
    calls = []

    def fake_probe_node(name):
        calls.append(name)
        return {"name": name, "alive": True, "gpus": []}

    monkeypatch.setattr(sch, "probe_node", fake_probe_node)
    nodes = sch.probe_all()
    by_name = {n["name"]: n for n in nodes}

    check("local node still probes normally", calls == ["local"], diag=f"calls={calls}")
    check("node001 marked down from shared route preflight",
          by_name["node001"]["alive"] is False and "timed out" in by_name["node001"]["error"],
          diag=str(by_name["node001"]))
    check("node002 reuses the same route failure without probing",
          by_name["node002"]["alive"] is False and "timed out" in by_name["node002"]["error"],
          diag=str(by_name["node002"]))


def test_probe_all_uses_normal_probe_when_outer_route_is_up(check, sch, monkeypatch):
    monkeypatch.setattr(sch, "NODES", {
        "node001": {
            "host": "202.197.46.16",
            "ssh_user": "zhengliang01",
            "ssh_proxy_jumps": ["jtl110gpu2", "jtl110gpu"],
            "sudo_ssh_host": "node001",
            "cpu_cores": 192,
        },
        "node002": {
            "host": "202.197.46.16",
            "ssh_user": "zhengliang01",
            "ssh_proxy_jumps": ["jtl110gpu2", "jtl110gpu"],
            "sudo_ssh_host": "node002",
            "cpu_cores": 192,
        },
    })
    monkeypatch.setattr(sch, "_probe_ssh_proxy_jump", lambda name, jump, timeout=8: True)
    monkeypatch.setattr(sch, "_probe_outer_ssh_route", lambda name: (True, ""))
    calls = []

    def fake_probe_node(name):
        calls.append(name)
        return {"name": name, "alive": True, "gpus": []}

    monkeypatch.setattr(sch, "probe_node", fake_probe_node)
    nodes = sch.probe_all()

    check("route-up path keeps per-node probing",
          sorted(calls) == ["node001", "node002"],
          diag=f"calls={calls}")
    check("all nodes are alive from fake probe",
          all(n["alive"] is True for n in nodes),
          diag=str(nodes))
