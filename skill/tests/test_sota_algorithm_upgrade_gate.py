from algorithm.experiments.sota_algorithm_upgrade_gate import build_sota_algorithm_upgrade_gate
from algorithm.experiments import sota_bridge_action_gate
from algorithm.experiments.sota_quadrant_pareto_gate import build_sota_quadrant_pareto_gate
from algorithm.theorem_dispatch.eta_lcb import (
    OnlineServiceEstimator,
    ReusableEtaCache,
    service_observation_key,
)
from algorithm.theorem_dispatch.global_dispatch import select_global_action
from algorithm.theorem_dispatch.state_service import StateDependentServiceCache
from simulation.defaults import build_default_cache
from simulation.fast_forward import ReplayPolicy, WorkloadSpec
from simulation.sota_baselines import SotaCandidateUnionPolicy


def test_sota_algorithm_upgrade_gate_closes_scoped_five_axis_certificate():
    report = build_sota_algorithm_upgrade_gate()

    assert report["status"] == "SOTA_ALGORITHM_UPGRADE_PASS"
    assert report["pass"] is True
    assert report["scoped_claim_ready"] is True
    assert report["strong_claim_ready"] is False
    assert report["adaptive_scalarized_single_policy"]["adaptive_scalarized_ready"] is True
    assert report["adaptive_scalarized_single_policy"]["not_pareto_dominated_all"] is True
    assert report["pareto_slack_fixed_online_policy"]["pareto_slack_ready"] is True
    assert report["pareto_slack_fixed_online_policy"]["not_pareto_dominated_all"] is True
    assert report["pareto_slack_fixed_online_policy"]["fixed_online_policy_not_pareto_dominated_all"] is True
    assert report["pareto_slack_fixed_online_policy"]["fixed_online_policy_beats_both_envelopes_all"] is True
    assert report["pareto_slack_fixed_online_policy"]["fixed_online_policy_pareto_dominates_sota_style_all"] is True
    assert report["state_dependent_marginal_cache"]["state_dependent_marginal_cache_ready"] is True
    assert report["global_batch_lookahead"]["lookahead_global_batch_ready"] is True
    assert report["eta_reuse"]["eta_reuse_ready"] is True
    assert report["expanded_sota_policy_families"]["expanded_sota_families_ready"] is True
    assert report["expanded_sota_policy_families"]["family_count"] >= 7
    assert report["sota_candidate_union_summary"]["union_mean_flow_beats_sota_envelope_all"] is True
    assert report["sota_candidate_union_summary"]["fixed_online_policy_not_pareto_dominated_all"] is True
    assert report["sota_candidate_union_summary"]["fixed_online_policy_beats_both_envelopes_all"] is True
    assert report["sota_candidate_union_summary"]["fixed_online_policy_pareto_dominates_sota_style_all"] is True


def test_sota_candidate_union_keeps_profile_trajectory_actions_distinct():
    cache = build_default_cache()
    spec = WorkloadSpec(
        workload_key="gpu_cnn_torch_resnet50",
        resource_kind="gpu_cnn",
        task_count=64,
        total_units=80,
        resource_count=2,
        variation_cv=0.05,
    )
    family_a = ReplayPolicy(name="family_a", fixed_profiles={spec.workload_key: 3})
    family_b = ReplayPolicy(name="family_b", fixed_profiles={spec.workload_key: 3})
    union = SotaCandidateUnionPolicy(
        name="unit_union",
        policy_families=(family_a, family_b),
        portfolio_specs=(spec,),
        selection_objective="makespan",
    )

    rows = union.candidate_rows(cache, spec)
    action_ids = {row["action_id"] for row in rows}

    assert len(rows) == 2
    assert action_ids == {
        "family_a|gpu_cnn_torch_resnet50|p3",
        "family_b|gpu_cnn_torch_resnet50|p3",
    }
    assert all(set(row["provenance"]) == {"family_a", "family_b"} for row in rows)


def test_sota_quadrant_gate_separates_tolerance_from_strict_dominance():
    report = build_sota_quadrant_pareto_gate()

    assert report["pass"] is True
    assert report["tolerance_pareto_ready"] is True
    assert report["strict_noninferiority_ready"] is True
    assert report["strict_pareto_dominance_ready"] is False
    q01 = next(row for row in report["quadrants"] if row["quadrant"] == "q01_low_cpu_high_gpu")
    assert q01["tolerance_pareto_ready"] is True
    assert q01["strict_noninferiority_ready"] is True
    assert q01["strict_pareto_dominance_ready"] is True


def test_sota_bridge_action_gate_keeps_real_probe_and_strict_claim_gated(monkeypatch):
    monkeypatch.setattr(
        sota_bridge_action_gate,
        "build_sota_strict_dominance_frontier",
        lambda: {
            "strict_pareto_ready": False,
            "within_tolerance_ready": True,
            "frontier_count": 2,
            "frontier": [],
        },
    )
    monkeypatch.setattr(
        sota_bridge_action_gate,
        "_phase_switch_search",
        lambda: {
            "current_cache_phase_switch_strict_ready": False,
            "targets": [
                {
                    "target": "q01_gpu_bound_cnn_resnet50",
                    "strict_ready": False,
                    "best_worst_ratio": 0.999,
                    "best_action": {"action_id": "bridge_cnn"},
                }
            ],
        },
    )
    monkeypatch.setattr(
        sota_bridge_action_gate,
        "_gpu_resource_audit",
        lambda **_: {
            "launch_ready": False,
            "safe_gpus": [],
            "safe_gpu_count": 0,
            "gpu_rows": [],
        },
    )

    report = sota_bridge_action_gate.build_sota_bridge_action_gate(
        allow_launch=True,
        run_id="unit_test_bridge_gate",
    )

    assert report["pass"] is True
    assert report["scoped_claim_ready"] is True
    assert report["strong_claim_ready"] is False
    assert report["strict_bridge_ready"] is False
    assert report["current_cache_phase_switch_strict_ready"] is False
    assert report["resource_launch_ready"] is False
    assert report["real_probe_launched"] is False
    assert report["status"] == "MEASURED_CACHE_EXTERNAL_POLICY_BRIDGE_WAIT_RESOURCE"


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
