import json

from skill import scheduler_state_compaction as state_compaction


def _verbose_neighbor(i: int) -> dict:
    return {
        "id": f"t{i}",
        "status": "running",
        "project": "KG-SYNTH",
        "signature": "very/long/signature/" + ("x" * 400),
        "node": "node006",
        "gpu_idx": 0,
        "started_at": 1000.0 + i,
        "elapsed_s": 60 + i,
        "eta_seconds": 120,
        "eta_source": "inline_eta",
        "progress_ratio": 0.25,
        "current_vram_mb": 100 + i,
        "peak_vram_mb": 200 + i,
        "current_ram_mb": 300 + i,
        "peak_ram_mb": 400 + i,
        "cpu_cores": 2,
        "ram_pressure_mb": 512,
        "evict_loss_protected": False,
        "last_progress_line": "Step 1/10 " + ("y" * 2000),
        "ckpt_dir": "/very/long/path/" + ("z" * 300),
    }


def test_save_state_compacts_persisted_resource_snapshots(sch):
    verbose_tasks = [_verbose_neighbor(i) for i in range(30)]
    bloated_snapshot = {
        "stage": "post_launch_estimated",
        "ts": 123.0,
        "target_node": "node006",
        "target_gpu_idx": 0,
        "gpu": {"used_mb": 12000, "total_mb": 24000, "over_one_third": True},
        "node_ram": {"free_ram_mb": 90000, "total_ram_mb": 192000},
        "task": _verbose_neighbor(999),
        "same_gpu_tasks": verbose_tasks,
        "same_node_running_tasks": verbose_tasks,
        "crossed_one_third_from_pre": True,
    }
    state = {
        "next_id": 2,
        "tasks": [{
            "id": "t1",
            "status": "running",
            "last_launch_pre_snapshot": bloated_snapshot,
            "last_launch_post_snapshot": bloated_snapshot,
            "last_resource_snapshot": bloated_snapshot,
            "last_crash_forensics": {
                "same_gpu_tasks": verbose_tasks,
                "last_resource_snapshot": bloated_snapshot,
            },
        }],
    }

    before = len(json.dumps(state, ensure_ascii=False))
    sch.save_state(state)
    after_state = sch.load_state()
    after = len(json.dumps(after_state, ensure_ascii=False))
    snap = after_state["tasks"][0]["last_resource_snapshot"]

    assert after < before / 5
    assert snap["same_gpu_task_count"] == 30
    assert len(snap["same_gpu_tasks"]) == 12
    assert snap["same_node_running_task_count"] == 30
    assert "same_node_running_tasks" not in snap
    assert "current_vram_mb" in snap["same_gpu_tasks"][0]
    assert "started_at" in snap["same_gpu_tasks"][0]
    assert "signature" not in snap["same_gpu_tasks"][0]
    assert "last_progress_line" not in snap["same_gpu_tasks"][0]
    assert "ckpt_dir" not in snap["task"]


def test_state_compaction_module_keeps_gpu_forensics_fields():
    snap = {
        "stage": "crash_probe",
        "same_gpu_tasks": [
            _verbose_neighbor(i) for i in range(20)
        ],
        "same_node_running_tasks": [
            _verbose_neighbor(i) for i in range(20)
        ],
    }

    compact = state_compaction.compact_resource_snapshot(snap)

    assert compact["same_gpu_task_count"] == 20
    assert len(compact["same_gpu_tasks"]) == 12
    assert compact["same_gpu_tasks"][-1]["id"] == "t19"
    assert "current_vram_mb" in compact["same_gpu_tasks"][-1]
    assert compact["same_node_running_task_count"] == 20
    assert "same_node_running_tasks" not in compact


def test_resource_snapshot_compaction_is_idempotent_for_neighbor_counts():
    snap = {
        "stage": "post_launch_estimated",
        "same_gpu_tasks": [_verbose_neighbor(i) for i in range(30)],
        "same_node_running_tasks": [_verbose_neighbor(i) for i in range(30)],
    }

    compact_once = state_compaction.compact_resource_snapshot(snap)
    compact_twice = state_compaction.compact_resource_snapshot(compact_once)

    assert compact_twice["same_gpu_task_count"] == 30
    assert len(compact_twice["same_gpu_tasks"]) == 12
    assert compact_twice["same_node_running_task_count"] == 30
    assert "same_node_running_tasks" not in compact_twice


def test_resource_snapshot_generation_is_compact(sch):
    tasks = []
    for i in range(15):
        tasks.append({
            "id": f"t{i}",
            "status": "running",
            "project": "KG-SYNTH",
            "signature": "long/signature/" + ("x" * 500),
            "node": "node006",
            "gpu_idx": 0,
            "started_at": 1000.0 + i,
            "eta_seconds": 30,
            "current_vram_mb": 100 + i,
            "peak_vram_mb": 200 + i,
            "current_ram_mb": 300 + i,
            "peak_ram_mb": 400 + i,
            "cpu_cores": 2,
            "last_progress_line": "Step 1/10 " + ("y" * 1000),
        })
    state = {"tasks": tasks}
    nodes = [{
        "name": "node006",
        "alive": True,
        "free_ram_mb": 100000,
        "total_ram_mb": 192000,
        "gpus": [{"idx": 0, "used_mb": 10000, "total_mb": 24000, "free_mb": 14000}],
    }]

    snap = sch._resource_snapshot_for_placement(state, nodes, tasks[-1], "node006", 0, "running_probe")

    assert snap["same_gpu_task_count"] == 15
    assert len(snap["same_gpu_tasks"]) == 12
    assert snap["same_node_running_task_count"] == 15
    assert "same_node_running_tasks" not in snap
    assert "signature" not in snap["same_gpu_tasks"][0]
    assert "last_progress_line" not in snap["same_gpu_tasks"][0]
    assert snap["same_gpu_tasks"][-1]["id"] == "t14"
