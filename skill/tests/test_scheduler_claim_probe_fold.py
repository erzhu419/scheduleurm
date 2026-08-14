from __future__ import annotations

from skill.scheduler_claim import probe_fold as fold


def _deps(snapshot, *, enabled=True, notifications=None):
    notifications = notifications if notifications is not None else []
    return fold.ClaimProbeFoldDeps(
        claim_enabled_for=lambda node: enabled,
        claim_snapshot=snapshot,
        notify=lambda *args, **kwargs: notifications.append((args, kwargs)),
    )


def test_fold_claims_into_probe_subtracts_budgeted_active_and_pending_claims():
    nodes = [{
        "name": "node001",
        "alive": True,
        "total_cpu": 16,
        "free_cpu": 12,
        "total_ram_mb": 100000,
        "free_ram_mb": 80000,
        "gpus": [{
            "idx": 0,
            "used_mb": 1000,
            "free_mb": 11000,
            "total_mb": 12000,
        }],
    }]
    claims = [
        {"task_id": "active", "pid": 123, "cpu_cores": 6, "ram_mb": 30000, "gpu_idx": 0, "vram_mb": 3000},
        {"task_id": "pending", "pid": None, "cpu_cores": 3, "ram_mb": 10000, "gpu_idx": 0, "vram_mb": 2000},
        {"task_id": "ignored-cpu", "pid": None, "ignore_cpu_capacity": True, "cpu_cores": 99, "ram_mb": 0},
    ]

    out = fold.fold_claims_into_probe(
        nodes,
        deps=_deps(lambda node: {"ok": True, "claims": claims, "intents": [{"task_id": "next"}]}),
    )

    node = out[0]
    assert node["free_cpu"] == 7
    assert node["free_ram_mb"] == 60000
    assert node["pending_claims"] == claims[1:]
    assert node["active_claims"] == claims[:1]
    assert node["claim_intents"] == [{"task_id": "next"}]
    assert node["claim_snapshot_error"] is None
    assert node["gpus"][0]["observed_used_mb"] == 1000
    assert node["gpus"][0]["observed_free_mb"] == 11000
    assert node["gpus"][0]["used_mb"] == 5000
    assert node["gpus"][0]["free_mb"] == 7000


def test_fold_claims_into_probe_notifies_and_keeps_node_on_snapshot_error():
    notifications = []
    nodes = [{"name": "node001", "alive": True, "free_cpu": 8, "gpus": []}]

    out = fold.fold_claims_into_probe(
        nodes,
        deps=_deps(lambda node: (_ for _ in ()).throw(RuntimeError("ssh down")),
                   notifications=notifications),
    )

    assert out == nodes
    assert notifications
    assert notifications[0][0][0] == "claims_probe_fold_error"
    assert notifications[0][0][1]["node"] == "node001"
