from __future__ import annotations

import json

from algorithm.experiments.or_algorithm_upgrade_gate import build_or_algorithm_upgrade_gate
from algorithm.theorem_dispatch.batch_policy import select_global_batch_placements
from algorithm.theorem_dispatch.eta_lcb import (
    OnlineServiceEstimator,
    ReusableEtaCache,
    service_observation_key,
)
from algorithm.theorem_dispatch.global_dispatch import select_global_action
from algorithm.theorem_dispatch.state_service import StateDependentServiceCache
from simulation.defaults import (
    build_default_cache,
    calibrated_backlog_aware_policy,
    calibrated_scalar_candidate_policy,
    legacy_policy,
)
from simulation.fast_forward import compare_policies
from simulation.tasksets import benchmark_tasksets


def test_global_dispatch_respects_multi_resource_conflicts():
    queue = {"w": 10.0}
    rows = [
        {
            "task_id": "a",
            "action_id": "a",
            "resource_ids": ("gpu0", "node0"),
            "lower_service": {"w": 2.0},
            "score_semantics": "robust_maxweight_lower_service",
            "theorem_ready": True,
        },
        {
            "task_id": "b",
            "action_id": "b",
            "resource_ids": ("gpu1", "node0"),
            "lower_service": {"w": 2.0},
            "score_semantics": "robust_maxweight_lower_service",
            "theorem_ready": True,
        },
    ]

    result = select_global_action(rows, queue, max_batch_size=2)

    assert result.candidate_configuration_count == 2
    assert len(result.selected_action_ids) == 1


def test_global_dispatch_lookahead_breaks_equal_score_toward_short_eta():
    queue = {"w": 3.0}
    rows = [
        {
            "task_id": "slow",
            "action_id": "a_slow",
            "resource_ids": ("gpu0",),
            "workload_key": "w",
            "lower_service": {"w": 1.0},
            "eta_lcb_s": 120.0,
            "score_semantics": "robust_maxweight_lower_service",
            "theorem_ready": True,
        },
        {
            "task_id": "fast",
            "action_id": "b_fast",
            "resource_ids": ("gpu1",),
            "workload_key": "w",
            "lower_service": {"w": 1.0},
            "eta_lcb_s": 10.0,
            "score_semantics": "robust_maxweight_lower_service",
            "theorem_ready": True,
        },
    ]

    without = select_global_action(rows, queue, max_batch_size=1, lookahead_weight=0.0)
    with_lookahead = select_global_action(rows, queue, max_batch_size=1, lookahead_weight=1.0)

    assert without.selected_action_ids == ("a_slow",)
    assert with_lookahead.selected_action_ids == ("b_fast",)


def test_state_dependent_cache_exposes_empty_and_resident_marginal_rows():
    state_cache = StateDependentServiceCache.load(base_cache=build_default_cache())

    gpu_empty = state_cache.lookup("marginal_cuda", 1, "empty", allow_base_fallback=False)
    gpu_resident = state_cache.lookup("marginal_cuda", 1, "high_vram_resident", allow_base_fallback=False)
    cpu_full = state_cache.lookup("marginal_cpu", 1, "cpu_full_resident", allow_base_fallback=False)

    assert gpu_empty.certified
    assert gpu_resident.certified
    assert cpu_full.certified
    assert gpu_resident.record.aggregate_rate > gpu_empty.record.aggregate_rate
    rows = state_cache.marginal_opportunity_rows("marginal_cuda")
    assert any(row["load_state"] == "high_vram_resident" for row in rows)
    assert all(row["certified"] for row in rows)


def test_online_lcb_eta_uses_dedup_key_and_positive_lower_bound():
    key = service_observation_key(
        workload_key="gpu_heavy_jax_matmul",
        profile=1,
        load_state="empty",
        node_bucket="unit-node:12gb",
        command_fingerprint="cmd",
        algorithm_mode="global_theorem_maxweight_v1",
    )
    estimator = OnlineServiceEstimator(confidence=0.80, min_samples=3)
    for rate in (9.8, 10.0, 10.2, 10.1, 9.9):
        estimator.add_progress(key=key, units_done=rate * 10.0, elapsed_s=10.0)

    estimate = estimator.estimate(key, remaining_units=100.0)

    assert "gpu_heavy_jax_matmul" in key
    assert "global_theorem_maxweight_v1" in key
    assert estimate.certified
    assert estimate.lcb_rate > 0.0
    assert estimate.eta_s and estimate.eta_s > 0.0


def test_reusable_eta_cache_reuses_identical_state_and_probes_new_state():
    key = service_observation_key(
        workload_key="gpu_heavy_jax_matmul",
        profile=1,
        load_state="high_vram_resident",
        node_bucket="node007:12gb",
        command_fingerprint="cmd",
        algorithm_mode="sota_adaptive_scalarized_v1",
    )
    cache = ReusableEtaCache(OnlineServiceEstimator(confidence=0.80, min_samples=3))
    for rate in (9.8, 9.9, 10.0, 9.7):
        cache.observe(key=key, units_done=rate * 10.0, elapsed_s=10.0)

    reused = cache.lookup(key, remaining_units=100.0)
    missing = cache.lookup(key.replace("|1|", "|2|"), remaining_units=100.0)

    assert reused.certified
    assert reused.reused_from_cache
    assert not reused.probe_required
    assert not missing.certified
    assert missing.probe_required


def test_backlog_aware_policy_improves_q01_without_regressing_cpu_or_cnn():
    cache = build_default_cache()
    for taskset_name in ("q01_gpu_bound_compute", "q01_gpu_bound_llm_inference", "q01_gpu_model_portfolio"):
        specs = benchmark_tasksets()[taskset_name].workload_specs(replayable_only=True, cache=cache)
        current = compare_policies(
            cache,
            specs,
            baseline=legacy_policy(),
            candidate=calibrated_scalar_candidate_policy(cache, specs),
            trials=21,
            seed=7,
        )
        upgraded = compare_policies(
            cache,
            specs,
            baseline=legacy_policy(),
            candidate=calibrated_backlog_aware_policy(),
            trials=21,
            seed=7,
        )
        assert upgraded.candidate.total_makespan_s <= current.candidate.total_makespan_s * 1.005
        assert upgraded.candidate.weighted_mean_flow_s <= current.candidate.weighted_mean_flow_s * 1.005

    for taskset_name in ("q01_gpu_bound_cnn_resnet50", "q10_cpu_host_bound", "q11_cpu_gpu_coupled"):
        specs = benchmark_tasksets()[taskset_name].workload_specs(replayable_only=True, cache=cache)
        current = compare_policies(
            cache,
            specs,
            baseline=legacy_policy(),
            candidate=calibrated_scalar_candidate_policy(cache, specs),
            trials=21,
            seed=7,
        )
        upgraded = compare_policies(
            cache,
            specs,
            baseline=legacy_policy(),
            candidate=calibrated_backlog_aware_policy(),
            trials=21,
            seed=7,
        )
        assert upgraded.candidate.total_makespan_s <= current.candidate.total_makespan_s * 1.005
        assert upgraded.candidate.weighted_mean_flow_s <= current.candidate.weighted_mean_flow_s * 1.005


def test_global_batch_policy_builds_scheduler_hint_only_certificate(tmp_path):
    queue_path = tmp_path / "queue.json"
    tasks = [
        {
            "id": "task-a",
            "status": "queued",
            "workload_key": "gpu_heavy_jax_matmul",
            "cmd": "python jax_matmul.py",
            "est_vram_mb": 512,
        },
        {
            "id": "task-b",
            "status": "queued",
            "workload_key": "hybrid_rl_resac_ant",
            "cmd": "python train.py --env Ant-v5",
            "est_vram_mb": 512,
        },
    ]
    queue_path.write_text(json.dumps({"tasks": tasks}), encoding="utf-8")
    nodes = [
        {
            "name": "unit-node",
            "alive": True,
            "gpus": [
                {"idx": 0, "used_mb": 100, "free_mb": 11900, "total_mb": 12000, "util_pct": 0, "running_task_count": 0},
                {"idx": 1, "used_mb": 100, "free_mb": 11900, "total_mb": 12000, "util_pct": 0, "running_task_count": 0},
            ],
        }
    ]

    result = select_global_batch_placements(tasks, nodes, context={"queue_file": str(queue_path)}, max_batch_size=2)

    assert result.scheduler_hook_ready
    assert set(result.placements) == {"task-a", "task-b"}


def test_scheduler_global_batch_hook_builds_soft_hint_plan(tmp_path, monkeypatch):
    monkeypatch.setenv("SCHEDULEURM_ALGORITHM", "global_theorem_maxweight_v1")
    import skill.scheduler as scheduler

    scheduler._configure_algorithm("global_theorem_maxweight_v1")
    queue_path = tmp_path / "queue.json"
    tasks = [
        {
            "id": "task-a",
            "status": "queued",
            "workload_key": "gpu_heavy_jax_matmul",
            "cmd": "python jax_matmul.py",
            "est_vram_mb": 512,
        },
        {
            "id": "task-b",
            "status": "queued",
            "workload_key": "hybrid_rl_resac_ant",
            "cmd": "python train.py --env Ant-v5",
            "est_vram_mb": 512,
        },
    ]
    queue_path.write_text(json.dumps({"tasks": tasks}), encoding="utf-8")
    monkeypatch.setattr(scheduler, "QUEUE_FILE", queue_path)
    nodes = [
        {
            "name": "unit-node",
            "alive": True,
            "gpus": [
                {"idx": 0, "used_mb": 100, "free_mb": 11900, "total_mb": 12000, "util_pct": 0, "running_task_count": 0},
                {"idx": 1, "used_mb": 100, "free_mb": 11900, "total_mb": 12000, "util_pct": 0, "running_task_count": 0},
            ],
        }
    ]

    plan, event = scheduler._algorithm_global_batch_plan(tasks, nodes)

    assert event["type"] == "algorithm_global_batch_plan"
    assert event["scheduler_hook_ready"] is True
    assert set(plan) == {"task-a", "task-b"}
    assert {row["gpu_idx"] for row in plan.values()} == {0, 1}


def test_or_algorithm_upgrade_gate_passes_scoped_claim():
    report = build_or_algorithm_upgrade_gate(seed=7)

    assert report["pass"] is True
    assert report["strong_claim_ready"] is False
    assert report["regression_count"] == 0
    assert report["improvement_count"] >= 3
