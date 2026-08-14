from skill.scheduler_cpu import plan as cpu_plan


def _nodes():
    return {
        "jtl110cpu": {"cpu_cores": 128, "logical_cpu": 256},
        "jtl110cpu2": {"cpu_cores": 128, "logical_cpu": 256},
    }


def test_cpu_worker_plan_matches_legacy_901_on_128():
    plan = cpu_plan.cpu_worker_plan_for_items(901, 128)

    assert plan["waves"] == 8
    assert plan["workers"] == 113
    assert plan["last_wave_items"] == 110


def test_cpu_batch_plan_splits_by_live_free_physical_cores():
    states = {
        "jtl110cpu": {"alive": True, "free_cpu": 64, "total_cpu": 128, "logical_cpu": 256},
        "jtl110cpu2": {"alive": True, "free_cpu": 128, "total_cpu": 128, "logical_cpu": 256},
    }

    plan = cpu_plan.cpu_batch_plan(
        901,
        ["jtl110cpu", "jtl110cpu2"],
        node_states=states,
        node_info_by_name=_nodes(),
        default_cpu_cores=1,
        windows_cpu_max_workers_per_process=0,
    )
    by_node = {row["node"]: row for row in plan}

    assert len(plan) == 2
    assert by_node["jtl110cpu"]["items"] == 300
    assert by_node["jtl110cpu2"]["items"] == 601
    assert by_node["jtl110cpu"]["workers"] == 60
    assert by_node["jtl110cpu2"]["workers"] == 121
    assert by_node["jtl110cpu"]["physical_cores"] == 64
    assert by_node["jtl110cpu"]["total_physical_cores"] == 128
    assert max(row["waves"] for row in plan) == 5


def test_cpu_batch_plan_skips_dead_or_full_nodes():
    states = {
        "jtl110cpu": {"alive": True, "free_cpu": 0, "total_cpu": 128, "logical_cpu": 256},
        "jtl110cpu2": {"alive": False, "free_cpu": 128, "total_cpu": 128, "logical_cpu": 256},
    }

    assert cpu_plan.cpu_batch_plan(
        901,
        ["jtl110cpu", "jtl110cpu2"],
        node_states=states,
        node_info_by_name=_nodes(),
    ) == []


def test_cpu_batch_plan_splits_windows_workers_into_lanes_when_needed():
    nodes = {
        "win": {
            "cpu_cores": 128,
            "os": "windows",
            "max_cpu_workers_per_process": 60,
        }
    }

    plan = cpu_plan.cpu_batch_plan(
        901,
        ["win"],
        node_info_by_name=nodes,
        windows_cpu_max_workers_per_process=64,
    )

    assert len(plan) == 2
    assert [row["worker_lane_index"] for row in plan] == [0, 1]
    assert {row["worker_lane_count"] for row in plan} == {2}
    assert sum(row["items"] for row in plan) == 901
    assert all(row["workers"] <= 60 for row in plan)
    assert all(row["unsplit_workers"] == 113 for row in plan)


def test_cpu_wave_summary_and_log_payload_keep_detailed_rows():
    states = {
        "jtl110cpu": {
            "alive": True,
            "free_cpu": 64,
            "total_cpu": 128,
            "logical_cpu": 256,
            "free_ram_mb": 400000,
            "total_ram_mb": 524288,
        },
        "jtl110cpu2": {
            "alive": True,
            "free_cpu": 128,
            "total_cpu": 128,
            "logical_cpu": 256,
            "free_ram_mb": 500000,
            "total_ram_mb": 524288,
        },
    }
    plan = cpu_plan.cpu_batch_plan(
        901,
        ["jtl110cpu", "jtl110cpu2"],
        node_states=states,
        node_info_by_name=_nodes(),
    )
    payload = cpu_plan.cpu_batch_log_payload(
        901,
        plan,
        node_states=states,
        templates={"cmd": "python eval.py"},
    )
    rows = {row["node"]: row for row in payload["nodes"]}

    assert payload["total_items"] == 901
    assert payload["node_count"] == 2
    assert rows["jtl110cpu"]["assigned_items"] == 300
    assert rows["jtl110cpu"]["free_physical_cores_used_for_plan"] == 64
    assert rows["jtl110cpu"]["live_node"]["logical_cpu"] == 256
    assert rows["jtl110cpu2"]["wave_plan"]["waves"] == rows["jtl110cpu2"]["waves"]
    assert cpu_plan.cpu_wave_summary(901, 113)["last_wave_items"] == 110


def test_cpu_parallel_template_values_and_worker_flag_rewrite():
    plan = {
        "node": "jtl110cpu",
        "start": 0,
        "end": 451,
        "items": 451,
        "workers": 113,
        "waves": 4,
        "physical_cores": 128,
        "last_wave_items": 112,
        "shard_index": 0,
        "num_shards": 2,
    }

    values = cpu_plan.cpu_parallel_template_values(plan, total_items=901, logical_items=39, item_multiplier=10)
    cmd = cpu_plan.rewrite_cpu_parallel_cmd(
        "python eval.py --start {start} --end {end} --workers auto "
        "--jobs={workers} --tag {node} --logical {logical_items} --mul {item_multiplier}",
        plan,
        total_items=901,
        logical_items=39,
        item_multiplier=10,
    )

    assert values["start"] == "0"
    assert values["total_items"] == "901"
    assert values["logical_items"] == "39"
    assert values["items_per_unit"] == "10"
    assert "--start 0" in cmd
    assert "--end 451" in cmd
    assert "--workers 113" in cmd
    assert "--jobs=113" in cmd
    assert "--tag jtl110cpu" in cmd
    assert "--logical 39" in cmd
    assert "--mul 10" in cmd


def test_cpu_parallel_env_keeps_zero_values():
    env = cpu_plan.cpu_parallel_env({
        "cpu_parallel_total_items": 901,
        "cpu_parallel_items": 451,
        "cpu_parallel_start": 0,
        "cpu_parallel_end": 451,
        "cpu_auto_workers": 113,
        "cpu_parallel_waves": 4,
        "cpu_parallel_physical_cores": 128,
        "cpu_parallel_shard_index": 0,
        "cpu_parallel_num_shards": 2,
    })

    assert env["SCHEDULEURM_CPU_SHARD_START"] == "0"
    assert env["SCHEDULEURM_CPU_SHARD_INDEX"] == "0"
    assert env["SCHEDULEURM_CPU_WORKERS"] == "113"
    assert env["SCHEDULEURM_CPU_TOTAL_ITEMS"] == "901"


def test_apply_cpu_parallel_plan_to_task_updates_task_and_rewrites_cmd():
    task = {
        "node": "node006",
        "cmd": "python eval.py --workers auto --slice {start}:{end}",
        "cpu_cores": 2,
        "cpu_parallel_items": 451,
        "cpu_parallel_total_items": 901,
        "cpu_parallel_start": 0,
        "cpu_parallel_end": 451,
        "cpu_parallel_shard_index": 0,
        "cpu_parallel_num_shards": 2,
        "cpu_batch_plan": {
            "workers": 113,
            "waves": 4,
            "physical_cores": 128,
            "total_physical_cores": 128,
            "last_wave_items": 112,
        },
    }

    plan = cpu_plan.apply_cpu_parallel_plan_to_task(
        task,
        node_state={"logical_cpu": 256},
        node_info={"cpu_cores": 128},
        default_cpu_cores=1,
    )

    assert plan["workers"] == 113
    assert task["cpu_auto_workers"] == 113
    assert task["cpu_cores"] == 113
    assert task["cpu_parallel_waves"] == 4
    assert task["cpu_parallel_last_wave_items"] == 112
    assert task["cmd"] == "python eval.py --workers 113 --slice 0:451"
    assert task["cpu_auto_worker_cmd_rewritten"] is True


def test_cpu_slot_floor_and_reservation_use_declared_worker_counts():
    assert cpu_plan.declared_cpu_slot_floor({
        "cpu_cores": 2,
        "cpu_declared_cores": 16,
        "cpu_auto_workers": 32,
        "cpu_batch_plan": {"workers": 64},
    }) == 64
    assert cpu_plan.reserved_cpu_slots_for_task(
        {"cpu_cores": 2, "cpu_auto_workers": 32},
        default_cpu_cores=8,
    ) == 32
    assert cpu_plan.reserved_cpu_slots_for_task({}, default_cpu_cores=8) == 8


def test_apply_cpu_slot_accounting_overlays_declared_load_on_slot_nodes():
    state = {
        "tasks": [
            {"id": "run", "status": "running", "node": "node006", "cpu_cores": 8},
            {"id": "launch", "status": "launching", "assigned_node": "node006", "cpu_cores": 1, "cpu_auto_workers": 170},
            {"id": "queued", "status": "queued", "node": "node006", "cpu_cores": 100},
            {"id": "other", "status": "running", "node": "node005", "cpu_cores": 40},
        ]
    }
    nodes = [
        {"name": "node006", "free_cpu": 168, "total_cpu": 192, "loadavg": 24.0},
        {"name": "node005", "free_cpu": 152, "total_cpu": 192, "loadavg": 40.0},
        {"name": "local", "free_cpu": 4, "total_cpu": 16, "loadavg": 12.0},
        {"name": "not-slot", "free_cpu": 9, "total_cpu": 10, "loadavg": 1.0},
    ]

    cpu_plan.apply_cpu_slot_accounting_to_nodes(
        state,
        nodes,
        node_configs={
            "node006": {"cpu_cores": 192, "cpu_slot_accounting": True},
            "node005": {"cpu_cores": 192},
            "local": {"cpu_cores": 16},
            "not-slot": {"cpu_cores": 10},
        },
        cpu_slot_accounting_node_names=["node006"],
        reserved_cpu_slots_for_task=lambda task: cpu_plan.reserved_cpu_slots_for_task(
            task,
            default_cpu_cores=1,
        ),
    )

    by_name = {node["name"]: node for node in nodes}
    assert by_name["node006"]["cpu_slot_reserved"] == 178
    assert by_name["node006"]["free_cpu"] == 14
    assert by_name["node006"]["cpu_slot_free"] == 14
    assert by_name["node006"]["loadavg"] == 24.0
    assert by_name["node006"]["observed_loadavg"] == 24.0
    assert by_name["node006"]["observed_free_cpu"] == 168
    assert by_name["node005"].get("cpu_slot_accounting") is None
    assert by_name["not-slot"].get("cpu_slot_accounting") is None
    assert by_name["local"]["cpu_slot_accounting"] is True
    assert by_name["local"]["free_cpu"] == 4
    assert by_name["local"]["loadavg"] == 12.0
