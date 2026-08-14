from __future__ import annotations

import json

from skill.scheduler_node.gates import (
    BlockedNodesDeps,
    NodeResourceGateDeps,
    blocked_nodes_for_task,
    hpc_cpu_pool_soft_require_nodes,
    launch_failed_nodes_for_task,
    node_gpu_task_block_reason,
    node_resources_ok,
    node_schedulable_cpu_free,
    task_requires_persistent_cpu_reservation,
)


def _pool():
    return [f"node{i:03d}" for i in range(1, 7)]


def test_blocked_nodes_matches_exact_and_recent_project(tmp_path):
    path = tmp_path / "escalations.jsonl"
    records = [
        {
            "task_id": "e1",
            "status": "pending",
            "category": "ENV_MISSING",
            "node": "node001",
            "signature": "sig-a",
            "cwd": "/other",
            "project": "BAPR",
            "ts": 0,
        },
        {
            "task_id": "e2",
            "status": "pending",
            "category": "PYTHON_IMPORT",
            "node": "node002",
            "signature": "sig-b",
            "cwd": "/elsewhere",
            "project": "BAPR",
            "cmd": "python3 train.py",
            "ts": 980,
        },
        {
            "task_id": "e3",
            "status": "pending",
            "category": "CUDA_RUNTIME",
            "node": "node003",
            "signature": "sig-c",
            "cwd": "/elsewhere",
            "project": "BAPR",
            "ts": 800,
        },
    ]
    path.write_text("\n".join(json.dumps(rec) for rec in records))

    deps = BlockedNodesDeps(
        escalations_file=path,
        project_wide_env_block_ttl_s=60,
        now=lambda: 1000.0,
    )

    blocked = blocked_nodes_for_task(
        {
            "signature": "sig-a",
            "cwd": "/new",
            "project": "BAPR",
            "cmd": "python3 train.py",
        },
        deps=deps,
    )

    assert blocked == {"node001", "node002"}


def test_hpc_cpu_pool_soft_require_nodes_recognizes_staged_cpu_families():
    transit = {
        "project": "TransitDuet",
        "require_node": "node003",
        "est_vram_mb": 0,
        "cmd": "python /home/zhengliang01/scheduleurm_work/TransitDuet/transit_duet/runner_v3.py",
    }
    kg = {
        "project": "KG-SYNTH",
        "est_vram_mb": 0,
        "cmd": "python benchmark_lodo_meta_prior.py",
    }
    gpu = dict(transit, est_vram_mb=1000)
    laplace = {
        "project": "Laplace-SMDP",
        "require_node": "node001",
        "est_vram_mb": 0,
        "cmd": "python experiments/run_planner_baseline_comparison.py",
    }

    assert hpc_cpu_pool_soft_require_nodes(transit) == _pool()
    assert hpc_cpu_pool_soft_require_nodes(kg) == _pool()
    assert hpc_cpu_pool_soft_require_nodes(laplace) == _pool()
    assert hpc_cpu_pool_soft_require_nodes(gpu) == []


def test_node_gpu_project_reservation_blocks_only_matching_gpu_tasks():
    node_info = {"blocked_gpu_projects": ["BAPR"]}

    assert node_gpu_task_block_reason(
        {"project": "BAPR", "est_vram_mb": 2300},
        "jtl311linux",
        node_info,
    ) == "gpu-policy block: project BAPR is reserved off this node"
    assert node_gpu_task_block_reason(
        {"project": "BAPR", "est_vram_mb": 0},
        "jtl311linux",
        node_info,
    ) is None
    assert node_gpu_task_block_reason(
        {"project": "other", "est_vram_mb": 2300},
        "jtl311linux",
        node_info,
    ) is None


def test_blocked_nodes_can_use_preloaded_snapshot_without_file_io(tmp_path):
    missing_path = tmp_path / "does-not-exist.jsonl"
    snapshot = {
        "e1": {
            "task_id": "e1",
            "status": "pending",
            "category": "ENV_MISSING",
            "node": "node003",
            "signature": "sig",
            "cwd": "/work",
            "project": "proj",
            "ts": 100.0,
        }
    }
    deps = BlockedNodesDeps(
        escalations_file=missing_path,
        project_wide_env_block_ttl_s=60,
        now=lambda: 100.0,
        latest_escalations=lambda: snapshot,
    )

    assert blocked_nodes_for_task(
        {"signature": "sig", "cwd": "/work", "project": "proj"},
        deps=deps,
    ) == {"node003"}


def test_stale_same_cwd_escalation_does_not_poison_new_signature(tmp_path):
    snapshot = {
        "e1": {
            "task_id": "e1",
            "status": "pending",
            "category": "ENV_MISSING",
            "node": "jtl110gpu",
            "signature": "old-signature",
            "cwd": "/shared/monorepo",
            "project": "BAPR",
            "ts": 100.0,
        }
    }
    deps = BlockedNodesDeps(
        escalations_file=tmp_path / "missing.jsonl",
        project_wide_env_block_ttl_s=60,
        now=lambda: 1000.0,
        latest_escalations=lambda: snapshot,
    )

    assert blocked_nodes_for_task(
        {
            "signature": "new-signature",
            "cwd": "/shared/monorepo",
            "project": "BAPR",
        },
        deps=deps,
    ) == set()
    assert blocked_nodes_for_task(
        {
            "signature": "old-signature",
            "cwd": "/shared/monorepo",
            "project": "BAPR",
        },
        deps=deps,
    ) == {"jtl110gpu"}


def test_python_import_block_does_not_cross_explicit_python_runtimes(tmp_path):
    snapshot = {
        "e1": {
            "task_id": "e1",
            "status": "pending",
            "category": "PYTHON_IMPORT",
            "node": "jtl110gpu2",
            "signature": "old-sig",
            "cwd": "/work",
            "project": "BAPR",
            "cmd": "python3 analyze.py",
            "ts": 100.0,
        }
    }
    deps = BlockedNodesDeps(
        escalations_file=tmp_path / "missing.jsonl",
        project_wide_env_block_ttl_s=3600,
        now=lambda: 101.0,
        latest_escalations=lambda: snapshot,
    )

    assert blocked_nodes_for_task(
        {
            "signature": "new-sig",
            "cwd": "/work",
            "project": "BAPR",
            "cmd": "/opt/resac/bin/python analyze.py",
        },
        deps=deps,
    ) == set()
    assert blocked_nodes_for_task(
        {
            "signature": "new-sig",
            "cwd": "/work",
            "project": "BAPR",
            "cmd": "python3 analyze.py",
        },
        deps=deps,
    ) == {"jtl110gpu2"}


def test_python_import_block_does_not_cross_python_entrypoints(tmp_path):
    snapshot = {
        "e1": {
            "task_id": "e1",
            "status": "pending",
            "category": "PYTHON_IMPORT",
            "node": "node001",
            "signature": "uniform-verifier-shard",
            "cwd": "/shared/kg",
            "project": "KG-SYNTH",
            "cmd": "/env/bin/python performance/run_uniform_verification_shard.py",
            "ts": 100.0,
        }
    }
    deps = BlockedNodesDeps(
        escalations_file=tmp_path / "missing.jsonl",
        project_wide_env_block_ttl_s=3600,
        now=lambda: 101.0,
        latest_escalations=lambda: snapshot,
    )

    assert blocked_nodes_for_task(
        {
            "signature": "transfer-matrix",
            "cwd": "/shared/kg",
            "project": "KG-SYNTH",
            "cmd": "/env/bin/python performance/benchmark_transfer_fairness.py",
        },
        deps=deps,
    ) == set()
    assert blocked_nodes_for_task(
        {
            "signature": "another-uniform-verifier-shard",
            "cwd": "/shared/kg",
            "project": "KG-SYNTH",
            "cmd": "/env/bin/python performance/run_uniform_verification_shard.py",
        },
        deps=deps,
    ) == {"node001"}


def test_launch_failed_nodes_expire_after_soft_block_ttl():
    task = {
        "launch_failed_nodes": {
            "node001": {"ts": 900.0, "error": "ssh transient"},
            "node002": {"ts": 950.0, "error": "ssh transient"},
            "missing": {"ts": 990.0, "error": "old node"},
        }
    }
    known = {"node001": {}, "node002": {}}

    assert launch_failed_nodes_for_task(
        task,
        known_nodes=known,
        soft_block_ttl_s=90,
        now=lambda: 1000.0,
    ) == {"node002"}


def test_launch_failed_nodes_without_ttl_preserve_legacy_soft_block():
    task = {"launch_failed_nodes": {"node001": {"ts": 1.0}}}

    assert launch_failed_nodes_for_task(
        task,
        known_nodes={"node001": {}},
        now=lambda: 1000.0,
    ) == {"node001"}


def test_node_schedulable_cpu_free_can_use_live_backfill():
    task = {"est_vram_mb": 0}
    state = {"free_cpu": 2, "observed_free_cpu": 48, "cpu_slot_accounting": True}
    info = {"live_cpu_backfill": True, "reserved_cpu_cores": 4}

    assert node_schedulable_cpu_free(task, state, info) == (44, "live", 2, 48)
    assert node_schedulable_cpu_free({"est_vram_mb": 500}, state, info) == (0, "slot", 2, 48)


def test_canonical_saas_uses_only_capped_live_surplus():
    task = {
        "est_vram_mb": 0,
        "cpu_declared_cores": 12,
        "cpu_cores": 12,
        "cmd": (
            "python benchmark_sota_fairness.py --method botorch_saasbo "
            "--saas-refit-schedule every_iteration --torch-device cpu"
        ),
    }
    state = {
        "free_cpu": 8,
        "observed_free_cpu": 96,
        "cpu_hard_free": 96,
        "cpu_slot_reserved": 184,
        "total_cpu": 192,
        "cpu_slot_accounting": True,
    }
    info = {
        "cpu_cores": 192,
        "live_cpu_backfill": True,
        "live_cpu_backfill_max_task_cores": 64,
        "persistent_cpu_live_backfill_max_declared_cores": 276,
        "reserved_cpu_cores": 20,
    }

    assert task_requires_persistent_cpu_reservation(task)
    assert node_schedulable_cpu_free(task, state, info) == (
        76, "persistent_live_capped", 8, 96,
    )


def test_canonical_saas_capped_live_surplus_stops_at_declared_ceiling():
    task = {
        "est_vram_mb": 0,
        "cpu_declared_cores": 12,
        "cpu_cores": 12,
        "cmd": (
            "python benchmark_sota_fairness.py --method botorch_saasbo "
            "--saas-refit-schedule every_iteration --torch-device cpu"
        ),
    }
    state = {
        "free_cpu": 0,
        "observed_free_cpu": 96,
        "cpu_hard_free": 96,
        "cpu_slot_reserved": 276,
        "total_cpu": 192,
        "cpu_slot_accounting": True,
    }
    info = {
        "cpu_cores": 192,
        "live_cpu_backfill": True,
        "live_cpu_backfill_max_task_cores": 64,
        "persistent_cpu_live_backfill_max_declared_cores": 276,
        "reserved_cpu_cores": 20,
    }

    assert node_schedulable_cpu_free(task, state, info) == (
        0, "persistent_live_capped", 0, 96,
    )


def test_canonical_saas_capped_live_surplus_obeys_live_headroom():
    task = {
        "est_vram_mb": 0,
        "cpu_declared_cores": 12,
        "cpu_cores": 12,
        "cmd": (
            "python benchmark_sota_fairness.py --method botorch_saasbo "
            "--saas-refit-schedule every_iteration --torch-device cpu"
        ),
    }
    state = {
        "free_cpu": 27,
        "observed_free_cpu": 24,
        "cpu_hard_free": 24,
        "cpu_slot_reserved": 165,
        "total_cpu": 192,
        "cpu_slot_accounting": True,
    }
    info = {
        "cpu_cores": 192,
        "live_cpu_backfill": True,
        "live_cpu_backfill_max_task_cores": 64,
        "persistent_cpu_live_backfill_max_declared_cores": 276,
        "reserved_cpu_cores": 20,
    }

    assert node_schedulable_cpu_free(task, state, info) == (
        4, "persistent_live_capped", 27, 24,
    )


def test_traffic_saas_keeps_declared_cpu_slots():
    task = {
        "est_vram_mb": 0,
        "cmd": "python performance/benchmark_traffic_final_contract.py",
    }

    assert task_requires_persistent_cpu_reservation(task)


def test_node_resources_ok_reports_primary_resource_gates():
    deps = NodeResourceGateDeps(
        default_cpu_cores=1,
        default_ram_mb=4096,
        hard_rule_bypassed=lambda *args, **kwargs: False,
        ignore_cpu_for_server_gpu_task=lambda task, node_state, node_info: False,
        task_is_gpu_capacity_task=lambda task: int(task.get("est_vram_mb") or 0) > 0,
        node_ram_headroom_mb=lambda node_state, node_info: int((node_info or {}).get("ram_headroom_mb") or 0),
        local_gpu_host_cpu_block_pct=90,
    )

    ok, why = node_resources_ok(
        {"est_vram_mb": 0, "cpu_cores": 4, "ram_mb": 1000},
        {"name": "node001", "free_cpu": 2, "total_cpu": 8, "free_ram_mb": 10000, "running_count": 0},
        {},
        deps=deps,
    )
    assert not ok
    assert why.startswith("cpu: need 4")

    ok, why = node_resources_ok(
        {"est_vram_mb": 0, "cpu_cores": 1, "ram_mb": 1000},
        {"name": "node001", "free_cpu": 8, "total_cpu": 8, "free_ram_mb": 10000, "running_count": 1},
        {"max_concurrent_running": 1},
        deps=deps,
    )
    assert not ok
    assert why.startswith("concurrency cap")

    ok, why = node_resources_ok(
        {"est_vram_mb": 1000, "cpu_cores": 1, "ram_mb": 1000},
        {
            "name": "node007",
            "free_cpu": 8,
            "total_cpu": 8,
            "free_ram_mb": 10000,
            "running_count": 0,
            "user_threads": 950,
            "user_thread_limit": 1000,
        },
        {"min_free_user_threads_for_gpu_task": 100},
        deps=deps,
    )
    assert not ok
    assert why.startswith("user thread pressure")

    ok, why = node_resources_ok(
        {
            "est_vram_mb": 0,
            "cpu_cores": 8,
            "ram_mb": 1000,
            "cmd": "JAX_PLATFORMS=cpu python -m eval",
        },
        {
            "name": "node006",
            "free_cpu": 100,
            "total_cpu": 192,
            "free_ram_mb": 10000,
            "running_count": 0,
            "user_threads": 3500,
            "user_thread_limit": 4096,
        },
        {"min_free_user_threads_for_jax_cpu_task": 1024},
        deps=deps,
    )
    assert not ok
    assert why.startswith("user thread pressure")

    ok, why = node_resources_ok(
        {
            "est_vram_mb": 0,
            "cpu_cores": 8,
            "ram_mb": 1000,
            "cmd": "python ordinary_cpu_job.py",
        },
        {
            "name": "node006",
            "free_cpu": 100,
            "total_cpu": 192,
            "free_ram_mb": 10000,
            "running_count": 0,
            "user_threads": 3500,
            "user_thread_limit": 4096,
        },
        {"min_free_user_threads_for_jax_cpu_task": 1024},
        deps=deps,
    )
    assert ok, why

    ok, why = node_resources_ok(
        {"est_vram_mb": 0, "cpu_cores": 1, "ram_mb": 9000},
        {"name": "node001", "free_cpu": 8, "total_cpu": 8, "free_ram_mb": 10000, "running_count": 0},
        {"ram_headroom_mb": 2048},
        deps=deps,
    )
    assert not ok
    assert "headroom 2048" in why
