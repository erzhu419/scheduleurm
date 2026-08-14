from __future__ import annotations

from skill.scheduler_placement_engine.post_dispatch_thresholds import (
    PostDispatchThresholdDeps,
    enforce_post_dispatch_thresholds,
)


def _task(task_id: str, **overrides):
    task = {
        "id": task_id,
        "status": "running",
        "started_at": 10.0,
        "node": "n1",
        "gpu_idx": 0,
        "cpu_cores": 1,
        "elapsed": 0,
        "est_vram_mb": 1000,
        "current_vram_mb": 0,
        "peak_vram_mb": 0,
    }
    task.update(overrides)
    return task


def _deps(
    evictions: list,
    notifications: list | None = None,
    *,
    node_configs: dict | None = None,
    cpu_high: bool = True,
    ram_snapshot: dict | None = None,
):
    notifications = notifications if notifications is not None else []

    def evict_to_queue(task, state, reason, kind):
        evictions.append((task["id"], reason, kind))
        task["status"] = "queued"

    def notify(event, payload, **kwargs):
        notifications.append((event, payload, kwargs))

    return PostDispatchThresholdDeps(
        node_configs=node_configs or {"local": {"cpu_cores": 4}, "n1": {}},
        default_cpu_cores=1,
        gpu_evict_stable_progress_min_age_s=600,
        gpu_evict_rollback_max_age_s=120,
        local_gpu_host_cpu_evict_pct=92,
        effective_elapsed_s=lambda task: float(task.get("elapsed") or 0),
        task_has_progress_evidence=lambda task: bool(task.get("progress")),
        task_progress_ratio=lambda task: task.get("progress_ratio"),
        task_ram_pressure_mb=lambda task: int(
            task.get("ram_pressure_mb") or task.get("ram_mb") or 0
        ),
        is_slurm_managed=lambda task: bool(task.get("slurm_job_id")),
        task_evict_loss_protected=lambda task: bool(
            task.get("loss_protected") or task.get("ckpt_protected")
        ),
        ckpt_task_evict_protected=lambda task: bool(task.get("ckpt_protected")),
        task_ignores_one_third_pack_rule=lambda task, info: bool(
            task.get("allow_gpu_over_one_third")
        ),
        local_cpu_pressure_high=lambda node, pct: (
            cpu_high,
            "cpu high" if cpu_high else "cpu ok",
        ),
        gpu_freeze_line_mb=lambda total: total // 3,
        gpu_threshold_snapshot=lambda gpu: dict(gpu),
        summarize_task_for_resource_log=lambda task: {
            "id": task.get("id"),
            "current_vram_mb": task.get("current_vram_mb", 0),
            "peak_vram_mb": task.get("peak_vram_mb", 0),
            "ram_pressure_mb": task.get("ram_pressure_mb", 0),
        },
        node_ram_snapshot=lambda node: ram_snapshot,
        format_mem_gb=lambda mb: f"{mb}MB",
        evict_to_queue=evict_to_queue,
        notify=notify,
        now=lambda: 123.0,
    )


def test_gpu_util_only_does_not_evict():
    evictions = []
    state = {"tasks": [_task("old"), _task("new", started_at=20.0)]}
    nodes = [{
        "name": "n1",
        "alive": True,
        "gpus": [{"idx": 0, "used_mb": 1000, "total_mb": 12000, "util_pct": 100}],
    }]

    evicted = enforce_post_dispatch_thresholds(state, nodes, deps=_deps(evictions))

    assert evicted == []
    assert evictions == []
    assert [task["status"] for task in state["tasks"]] == ["running", "running"]


def test_gpu_freeze_line_evicts_newest_no_progress_and_adjusts_vram():
    evictions = []
    old = _task("old", started_at=10.0, current_vram_mb=1200, peak_vram_mb=1300)
    new = _task("new", started_at=25.0, current_vram_mb=4800, peak_vram_mb=5000)
    state = {"tasks": [old, new]}
    nodes = [{
        "name": "n1",
        "alive": True,
        "gpus": [{"idx": 0, "used_mb": 6000, "total_mb": 12000, "util_pct": 20}],
    }]

    evicted = enforce_post_dispatch_thresholds(state, nodes, deps=_deps(evictions))

    assert evicted == ["new"]
    assert evictions[0][0] == "new"
    assert evictions[0][2] == "gpu_one_third"
    assert new["status"] == "queued"
    assert new["est_vram_mb"] == 5750
    assert new["est_vram_mb_adjusted_by_eviction"]["observed_vram_mb"] == 5000
    assert new["last_resource_eviction"]["same_gpu_tasks"][1]["id"] == "new"


def test_stable_progress_task_protected_on_gpu():
    evictions = []
    stable = _task(
        "stable",
        started_at=5.0,
        elapsed=900,
        progress=True,
        progress_ratio=0.7,
    )
    fresh = _task("fresh", started_at=50.0, elapsed=20)
    state = {"tasks": [stable, fresh]}
    nodes = [{
        "name": "n1",
        "alive": True,
        "gpus": [{"idx": 0, "used_mb": 6000, "total_mb": 12000, "util_pct": 30}],
    }]

    evicted = enforce_post_dispatch_thresholds(state, nodes, deps=_deps(evictions))

    assert evicted == ["fresh"]
    assert stable["status"] == "running"
    assert fresh["status"] == "queued"
    assert fresh["last_resource_eviction"]["protected_stable_progress_task_ids"] == ["stable"]


def test_ram_headroom_picks_ram_pressure_and_protects_ckpt():
    evictions = []
    protected = _task(
        "ckpt",
        gpu_idx=None,
        ram_pressure_mb=16000,
        ckpt_protected=True,
    )
    heavy = _task("heavy", gpu_idx=None, ram_pressure_mb=12000)
    light = _task("light", gpu_idx=None, ram_pressure_mb=1000, started_at=100.0)
    state = {"tasks": [protected, heavy, light]}
    nodes = [{"name": "n1", "alive": True, "gpus": []}]
    ram = {
        "below_eviction_headroom": True,
        "free_mb": 256,
        "eviction_headroom_mb": 4096,
        "headroom_mb": 4608,
        "headroom_grace_mb": 512,
    }

    evicted = enforce_post_dispatch_thresholds(
        state, nodes, deps=_deps(evictions, ram_snapshot=ram)
    )

    assert evicted == ["heavy"]
    assert heavy["status"] == "queued"
    assert protected["status"] == "running"
    payload = heavy["last_resource_eviction"]
    assert payload["kind"] == "node_ram_headroom"
    assert payload["protected_evict_loss_task_ids"] == ["ckpt"]
    assert payload["protected_ckpt_task_ids"] == ["ckpt"]


def test_local_cpu_budget_not_evict_when_host_cpu_not_high():
    evictions = []
    notifications = []
    state = {
        "tasks": [
            _task("a", node="local", gpu_idx=None, cpu_cores=2),
            _task("b", node="local", gpu_idx=None, cpu_cores=2),
        ]
    }
    nodes = [{"name": "local", "alive": True, "total_cpu": 2, "gpus": []}]

    evicted = enforce_post_dispatch_thresholds(
        state,
        nodes,
        deps=_deps(
            evictions,
            notifications,
            node_configs={"local": {"cpu_cores": 2}},
            cpu_high=False,
        ),
    )

    assert evicted == []
    assert evictions == []
    assert notifications[0][0] == "local_cpu_over_reserved_no_evict"
    assert notifications[0][1]["reserved"] == 4
    assert [task["status"] for task in state["tasks"]] == ["running", "running"]


def test_local_cpu_budget_evicts_until_under_budget_when_host_cpu_high():
    evictions = []
    tasks = [
        _task("old", node="local", gpu_idx=None, cpu_cores=2, started_at=1.0),
        _task("mid", node="local", gpu_idx=None, cpu_cores=2, started_at=2.0),
        _task("new", node="local", gpu_idx=None, cpu_cores=2, started_at=3.0),
    ]
    state = {"tasks": tasks}
    nodes = [{"name": "local", "alive": True, "total_cpu": 4, "gpus": []}]

    evicted = enforce_post_dispatch_thresholds(
        state,
        nodes,
        deps=_deps(evictions, node_configs={"local": {"cpu_cores": 4}}, cpu_high=True),
    )

    assert evicted == ["new"]
    assert evictions[0][2] == "local_cpu_budget"
    assert tasks[2]["status"] == "queued"
    assert tasks[0]["status"] == "running"
    assert tasks[1]["status"] == "running"


def test_slurm_and_auto_adopted_are_ineligible_for_gpu_eviction():
    evictions = []
    state = {
        "tasks": [
            _task("slurm", slurm_job_id=42, current_vram_mb=3000),
            _task("adopted", auto_adopted=True, started_at=20.0, current_vram_mb=3000),
        ]
    }
    nodes = [{
        "name": "n1",
        "alive": True,
        "gpus": [{"idx": 0, "used_mb": 6000, "total_mb": 12000, "util_pct": 20}],
    }]

    evicted = enforce_post_dispatch_thresholds(state, nodes, deps=_deps(evictions))

    assert evicted == []
    assert evictions == []
    assert [task["status"] for task in state["tasks"]] == ["running", "running"]


def test_measurement_reserved_node_skips_post_dispatch_eviction():
    evictions = []
    task = _task("keep-running", current_vram_mb=6000, peak_vram_mb=6000)
    state = {"tasks": [task]}
    nodes = [{
        "name": "n1",
        "alive": True,
        "measurement_reservation_active": True,
        "free_ram_mb": 0,
        "gpus": [{"idx": 0, "used_mb": 12000, "total_mb": 12000, "util_pct": 100}],
    }]

    assert enforce_post_dispatch_thresholds(state, nodes, deps=_deps(evictions)) == []
    assert evictions == []
    assert task["status"] == "running"
