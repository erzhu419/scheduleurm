from __future__ import annotations

import pytest


def test_new_launch_routing_ignores_removed_backend_config_and_task_knobs(sch, monkeypatch):
    monkeypatch.setattr(
        sch,
        "NODES",
        {"node001": {"host": "node001", "slurm_backend": "slurm"}},
        raising=False,
    )
    monkeypatch.setattr(
        sch,
        "run_on",
        lambda *args, **kwargs: pytest.fail("Removed backend capability probe should not run"),
        raising=False,
    )

    backend = sch.HybridBackend()
    task = {"id": "t1", "node": "node001", "est_vram_mb": 16000, "slurm_partition": "gpu"}

    assert backend.requires_local_capacity_check("node001", task) is True
    assert isinstance(backend._backend_for("node001", task), sch.LocalBackend)


def test_legacy_slurm_job_records_route_to_readonly_legacy_backend(sch):
    backend = sch.HybridBackend()

    assert isinstance(
        backend._backend_for_task({"id": "old", "node": "node001", "slurm_job_id": 123}),
        sch.LegacyExternalBackend,
    )
    assert not hasattr(sch, "SlurmBackend")


def test_legacy_external_backend_never_submits_or_probes_jobs(sch):
    backend = sch.LegacyExternalBackend()
    task = {"id": "old", "status": "running", "node": "node001", "slurm_job_id": 123}

    ok, msg = backend.launch({"id": "new", "node": "node001"})
    assert ok is False
    assert "Legacy external scheduler records are read-only" in msg

    ok, msg = backend.kill(task)
    assert ok is False
    assert "Legacy external scheduler records are read-only" in msg

    result = backend.batch_probe({"tasks": [task]})
    assert result["old"]["state"] == "unknown"
    assert "Legacy external scheduler records are read-only" in result["old"]["error"]


def test_deprecated_external_submit_fields_do_not_block_local_placement(sch, monkeypatch):
    monkeypatch.setattr(
        sch,
        "NODES",
        {
            "node001": {
                "host": "node001",
                "cpu_cores": 8,
                "ram_mb": 64000,
                "max_vram_per_task": None,
            }
        },
        raising=False,
    )
    task = {
        "id": "tDeprecatedBackendCompat",
        "status": "queued",
        "priority": "normal",
        "project": "unit",
        "signature": "unit/deprecated-backend-compat",
        "est_vram_mb": 1000,
        "ram_mb": 1000,
        "cpu_cores": 1,
        "slurm_partition": "gpu",
    }
    nodes = [{
        "name": "node001",
        "alive": True,
        "free_cpu": 8,
        "free_ram_mb": 64000,
        "total_ram_mb": 64000,
        "running_count": 0,
        "gpus": [{"idx": 0, "used_mb": 0, "free_mb": 24000, "total_mb": 24000, "util_pct": 0}],
    }]

    assert sch.pick_placement(task, nodes) == ("node001", 0)
