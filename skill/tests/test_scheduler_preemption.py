from __future__ import annotations

from skill.scheduler_placement_engine.preemption import PreemptionDeps, preempt_for_high_priority


NOW = 10_000.0


def _queued_high(task_id="hi", **overrides):
    task = {
        "id": task_id,
        "status": "queued",
        "priority": "high",
        "submitted_at": NOW - 20 * 60,
        "require_node": "n1",
        "cpu_cores": 6,
        "ram_mb": 6000,
    }
    task.update(overrides)
    return task


def _victim(task_id, **overrides):
    task = {
        "id": task_id,
        "status": "running",
        "priority": "normal",
        "node": "n1",
        "started_at": NOW - 30 * 60,
        "cpu_cores": 2,
        "ram_mb": 2000,
    }
    task.update(overrides)
    return task


def _deps(evictions, *, protected=None, slurm=None, ignore_cpu=False, now=NOW, max_victims=3):
    protected = protected or set()
    slurm = slurm or set()

    def evict_to_queue(victim, state, reason, kind):
        evictions.append((victim["id"], reason, kind))
        victim["status"] = "queued"

    return PreemptionDeps(
        node_configs={"n1": {"ram_headroom_mb": 0}, "n2": {"ram_headroom_mb": 0}},
        queue_wait_min=5,
        victim_min_age_min=10,
        victim_max_age_min=240,
        max_victims_per_dispatch=max_victims,
        default_cpu_cores=1,
        default_ram_mb=1000,
        node_ram_headroom_mb=lambda node_state, node_info: 0,
        ignore_cpu_for_server_gpu_task=lambda *args, **kwargs: ignore_cpu,
        is_slurm_managed=lambda task: task["id"] in slurm,
        task_evict_loss_protected=lambda task: task["id"] in protected,
        task_ram_pressure_mb=lambda task: int(task.get("ram_mb") or 0),
        task_progress_ratio=lambda task: task.get("progress"),
        evict_to_queue=evict_to_queue,
        format_mem_gb=lambda mb: f"{mb}MB",
        now=lambda: now,
    )


def test_preempt_evicts_until_cpu_and_ram_deficit_are_covered():
    evictions = []
    state = {
        "tasks": [
            _queued_high(),
            _victim("v1", started_at=NOW - 50 * 60),
            _victim("v2", started_at=NOW - 40 * 60),
            _victim("v3", started_at=NOW - 30 * 60),
        ]
    }
    nodes = [{"name": "n1", "free_cpu": 0, "free_ram_mb": 0}]

    evicted = preempt_for_high_priority(state, nodes, deps=_deps(evictions))

    assert [event["id"] for event in evicted] == ["v3", "v2", "v1"]
    assert all(event["target_id"] == "hi" for event in evicted)
    assert evictions[0][2] == "preempt_high_priority"
    assert "deficit cpu=6, ram=6000MB" in evictions[0][1]
    assert all(task["status"] == "queued" for task in state["tasks"][1:])


def test_preempt_skips_ineligible_victims_and_slurm_managed_tasks():
    evictions = []
    state = {
        "tasks": [
            _queued_high(cpu_cores=2, ram_mb=2000),
            _victim("fresh", started_at=NOW - 5 * 60),
            _victim("old", started_at=NOW - 300 * 60),
            _victim("adopted", auto_adopted=True),
            _victim("slurm"),
            _victim("ok"),
        ]
    }
    nodes = [{"name": "n1", "free_cpu": 0, "free_ram_mb": 0}]

    evicted = preempt_for_high_priority(
        state,
        nodes,
        deps=_deps(evictions, slurm={"slurm"}),
    )

    assert [event["id"] for event in evicted] == ["ok"]
    assert evictions == [("ok", evictions[0][1], "preempt_high_priority")]
    assert state["tasks"][4]["status"] == "running"


def test_preempt_prefers_unprotected_victim_but_falls_back_if_all_protected():
    evictions = []
    state = {
        "tasks": [
            _queued_high(cpu_cores=2, ram_mb=1000),
            _victim("protected", ram_mb=9000),
            _victim("safe", ram_mb=1000),
        ]
    }
    nodes = [{"name": "n1", "free_cpu": 0, "free_ram_mb": 0}]

    evicted = preempt_for_high_priority(
        state,
        nodes,
        deps=_deps(evictions, protected={"protected"}),
    )

    assert [event["id"] for event in evicted] == ["safe"]
    assert evicted[0]["protected_skipped"] == ["protected"]

    evictions.clear()
    state = {"tasks": [_queued_high(cpu_cores=2, ram_mb=1000), _victim("only", ram_mb=9000)]}
    evicted = preempt_for_high_priority(
        state,
        nodes,
        deps=_deps(evictions, protected={"only"}),
    )
    assert [event["id"] for event in evicted] == ["only"]
    assert evicted[0]["protected_skipped"] == ["only"]


def test_preempt_uses_ram_headroom_and_ignore_cpu_switch():
    evictions = []
    state = {"tasks": [_queued_high(cpu_cores=99, ram_mb=5000), _victim("ram", ram_mb=4000)]}
    nodes = [{"name": "n1", "free_cpu": 0, "free_ram_mb": 1000}]

    evicted = preempt_for_high_priority(
        state,
        nodes,
        deps=_deps(evictions, ignore_cpu=True),
    )

    assert [event["id"] for event in evicted] == ["ram"]
    assert evicted[0]["cpu_deficit"] == 0
    assert evicted[0]["ram_deficit"] == 4000


def test_preempt_only_processes_one_high_priority_task_per_node_per_dispatch():
    evictions = []
    state = {
        "tasks": [
            _queued_high("hi1", submitted_at=NOW - 30 * 60, cpu_cores=2, ram_mb=1000),
            _queued_high("hi2", submitted_at=NOW - 25 * 60, cpu_cores=2, ram_mb=1000),
            _victim("v1"),
            _victim("v2"),
        ]
    }
    nodes = [{"name": "n1", "free_cpu": 0, "free_ram_mb": 0}]

    evicted = preempt_for_high_priority(state, nodes, deps=_deps(evictions))

    assert [event["target_id"] for event in evicted] == ["hi1"]
    assert len(evicted) == 1


def test_preempt_skips_measurement_reserved_node():
    evictions = []
    state = {"tasks": [_queued_high(), _victim("keep-running")]}
    nodes = [{
        "name": "n1",
        "free_cpu": 0,
        "free_ram_mb": 0,
        "measurement_reservation_active": True,
    }]

    assert preempt_for_high_priority(state, nodes, deps=_deps(evictions)) == []
    assert evictions == []
    assert state["tasks"][1]["status"] == "running"
