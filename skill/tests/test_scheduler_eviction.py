from skill.scheduler_placement_engine.eviction import (
    EvictionDeps,
    evict_node_cooldown_block_reason,
    evict_to_queue,
    eviction_cooldown_block_reason,
)


def _deps(*, now=1000.0, cpu_high=True, calls=None):
    calls = calls if calls is not None else []

    def release(task):
        calls.append(("release", task.get("node")))

    def task_pids(task):
        calls.append(("pids", task.get("node")))
        return task.get("remote_pids") or []

    def actor(task, kind, reason):
        calls.append(("actor", task.get("node"), kind, reason))
        return {"action": kind}

    def kill(task, **kwargs):
        calls.append(("kill", task.get("node"), kwargs))
        return True, "killed"

    def notify(event, payload, **kwargs):
        calls.append(("notify", event, payload, kwargs))

    def set_usage(task, vram, ram, pcpu):
        calls.append(("usage", task.get("node"), vram, ram, pcpu))
        task["current_vram_mb"] = vram
        task["current_ram_mb"] = ram
        task["current_pcpu"] = pcpu

    def clear_eta(task, **kwargs):
        calls.append(("clear_eta", task.get("node"), kwargs))
        task.pop("eta_seconds", None)

    return EvictionDeps(
        release_task_claims_and_intents=release,
        task_pids=task_pids,
        record_task_kill_actor=actor,
        kill_task_processes=kill,
        notify=notify,
        set_current_usage=set_usage,
        clear_live_eta_fields=clear_eta,
        local_cpu_pressure_high=lambda node_state, threshold: (cpu_high, "cpu"),
        evict_relaunch_cooldown_s=1800,
        local_cpu_evict_node_cooldown_s=600,
        local_gpu_host_cpu_block_pct=85,
        now=lambda: now,
    )


def test_evict_to_queue_releases_claim_before_clearing_node_and_sets_node_cooldown():
    calls = []
    task = {
        "id": "t1",
        "status": "running",
        "node": "node001",
        "gpu_idx": 0,
        "process_group": 123,
        "log_path": "/tmp/log",
        "started_at": 900.0,
        "finished_at": None,
        "_diagnosis": {"x": 1},
        "remote_pids": [11, 12],
        "alive_pids": [11],
        "peak_vram_mb": 123,
        "peak_ram_mb": 456,
        "eta_seconds": 99,
    }

    evict_to_queue(
        task,
        {"tasks": [task]},
        "making room",
        "preempt",
        deps=_deps(calls=calls),
    )

    assert calls[0] == ("release", "node001")
    assert calls[1] == ("pids", "node001")
    assert calls[2][0:3] == ("actor", "node001", "preempt")
    assert calls[3] == ("kill", "node001", {"timeout": 15})
    assert task["status"] == "queued"
    assert task["node"] is None
    assert task["gpu_idx"] is None
    assert task["remote_pids"] == []
    assert task["alive_pids"] == []
    assert task["peak_vram_mb"] == 0
    assert task["peak_ram_mb"] == 0
    assert task["last_block_reason"] == "making room"
    assert task["last_evicted_at"] == 1000.0
    assert task["last_eviction_kind"] == "preempt"
    assert task["evict_node_cooldowns"]["node001"] == 2800.0
    assert "eta_seconds" not in task


def test_evict_to_queue_uses_shorter_local_cpu_budget_cooldown():
    task = {"id": "t1", "status": "running", "node": "local", "remote_pids": [1]}

    evict_to_queue(
        task,
        {"tasks": [task]},
        "cpu pressure",
        "local_cpu_budget",
        deps=_deps(now=1000.0),
    )

    assert task["evict_node_cooldowns"]["local"] == 1600.0


def test_legacy_global_cooldown_migrates_to_node_specific_when_resource_eviction_has_node():
    task = {
        "evict_cooldown_until": 1500.0,
        "last_resource_eviction": {"node": "node006"},
        "last_eviction_kind": "gpu_one_third",
    }

    reason = eviction_cooldown_block_reason(task, now=1000.0, deps=_deps())

    assert reason == ""
    assert "evict_cooldown_until" not in task
    assert task["evict_node_cooldowns"]["node006"] == 1500.0


def test_global_cooldown_without_node_still_blocks_until_expiry():
    task = {"evict_cooldown_until": 1500.0, "last_eviction_kind": "preempt"}

    reason = eviction_cooldown_block_reason(task, now=1000.0, deps=_deps())

    assert "cooling down 500s" in reason
    assert task["evict_cooldown_until"] == 1500.0
    assert eviction_cooldown_block_reason(task, now=1600.0, deps=_deps()) == ""
    assert "evict_cooldown_until" not in task


def test_node_cooldown_blocks_only_matching_node_and_expires():
    task = {
        "evict_node_cooldowns": {"node001": 1500.0},
        "last_eviction_kind": "preempt",
    }

    assert evict_node_cooldown_block_reason(task, "node002", now=1000.0, deps=_deps()) == ""
    assert "cooling down 500s" in evict_node_cooldown_block_reason(
        task, "node001", now=1000.0, deps=_deps())
    assert evict_node_cooldown_block_reason(task, "node001", now=1600.0, deps=_deps()) == ""
    assert "evict_node_cooldowns" not in task


def test_local_cpu_cooldown_is_cleared_when_cpu_pressure_drops():
    task = {
        "last_eviction_kind": "local_cpu_budget",
        "last_evicted_at": 1000.0,
    }

    reason = evict_node_cooldown_block_reason(
        task,
        "local",
        now=1100.0,
        node_state={"name": "local"},
        deps=_deps(now=1100.0, cpu_high=False),
    )

    assert reason == ""
    assert "evict_node_cooldowns" not in task
