from algorithm.experiments.sota_algorithm_upgrade_gate import build_sota_algorithm_upgrade_gate
from algorithm.theorem_dispatch.eta_lcb import (
    OnlineServiceEstimator,
    ReusableEtaCache,
    service_observation_key,
)
from algorithm.theorem_dispatch.global_dispatch import select_global_action
from algorithm.theorem_dispatch.state_service import StateDependentServiceCache
from simulation.defaults import build_default_cache


def test_sota_algorithm_upgrade_gate_closes_scoped_five_axis_certificate():
    report = build_sota_algorithm_upgrade_gate()

    assert report["status"] == "SOTA_ALGORITHM_UPGRADE_PASS"
    assert report["pass"] is True
    assert report["scoped_claim_ready"] is True
    assert report["strong_claim_ready"] is False
    assert report["adaptive_scalarized_single_policy"]["adaptive_scalarized_ready"] is True
    assert report["adaptive_scalarized_single_policy"]["not_pareto_dominated_all"] is True
    assert report["state_dependent_marginal_cache"]["state_dependent_marginal_cache_ready"] is True
    assert report["global_batch_lookahead"]["lookahead_global_batch_ready"] is True
    assert report["eta_reuse"]["eta_reuse_ready"] is True
    assert report["expanded_sota_policy_families"]["expanded_sota_families_ready"] is True
    assert report["expanded_sota_policy_families"]["family_count"] >= 7
    assert report["sota_candidate_union_summary"]["union_makespan_beats_sota_envelope_all"] is True
    assert report["sota_candidate_union_summary"]["union_mean_flow_beats_sota_envelope_all"] is True


def test_global_dispatch_lookahead_is_bounded_tie_breaker():
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

    no_lookahead = select_global_action(rows, {"w": 3.0}, max_batch_size=1, lookahead_weight=0.0)
    with_lookahead = select_global_action(rows, {"w": 3.0}, max_batch_size=1, lookahead_weight=1.0)

    assert no_lookahead.selected_action_ids == ("a_slow",)
    assert with_lookahead.selected_action_ids == ("b_fast",)


def test_state_dependent_cache_exposes_resident_marginal_service():
    state_cache = StateDependentServiceCache.load(base_cache=build_default_cache())

    gpu_empty = state_cache.lookup("marginal_cuda", 1, "empty", allow_base_fallback=False)
    gpu_resident = state_cache.lookup("marginal_cuda", 1, "high_vram_resident", allow_base_fallback=False)
    cpu_full = state_cache.lookup("marginal_cpu", 1, "cpu_full_resident", allow_base_fallback=False)
    rows = state_cache.marginal_opportunity_rows("marginal_cuda")

    assert gpu_empty.certified
    assert gpu_resident.certified
    assert cpu_full.certified
    assert gpu_resident.record.aggregate_rate > gpu_empty.record.aggregate_rate
    assert any(row["load_state"] == "high_vram_resident" for row in rows)


def test_reusable_eta_cache_reuses_certified_identical_state_and_probes_unknown_state():
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
