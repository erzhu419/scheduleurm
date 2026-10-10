"""Unknown-rate confirmation needs unobserved seeds and frozen causal rules."""
from copy import deepcopy

import pytest

import algorithm.experiments.aligned_hardware_observed_rate_age_tail_confirmation as confirmation


def _development(monkeypatch, tmp_path):
    monkeypatch.setattr(confirmation, "ROOT", tmp_path)
    monkeypatch.setattr(confirmation, "DEVELOPMENT", tmp_path / "md/experiment_artifacts/observed_development")
    sources = {"algorithm/observed_core.py": "frozen causal core", confirmation.SOURCE: "confirmation driver"}
    for path, source in sources.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
    models = [{"node": f"node{i}", "lanes": 4 if i == 1 else 2} for i in range(4)]
    development = {
        "registered_at_utc": "observed-dev-frozen", "candidate_method": "causal_count_rate_age_tail",
        "models": models, "loads": [.8, .95], "arrival_seeds": [41, 42, 43],
        "tapes": [{"seed": 41, "arrivals": [[99., 0, 0]]}],
        "capacities": [{"node": m["node"], "lanes": m["lanes"], "load": load,
                        "arrival_rates": [load]} for m in models for load in [.8, .95]],
        "arrival_horizon": 600., "observation_end": 1200.,
        "execution_order": [[m["node"], m["lanes"]] for m in models],
        "performance": {"min_gain": .01, "group_secondary_tolerance": .05},
        "planning_samples": 8, "planning_event_cap": 128,
        "planning_seed_rule": "arrival-plan-v1:{node}:{load}:online", "clock": "frozen clock",
        "labels": ["maxweight", "arrival_rollout"], "method": "frozen observed-rate method",
        "information": "true rate belongs only to environment", "budget": "frozen budget",
        "rate_estimator": {"rule": "counts/min(now,H)", "includes_admitted_history": True},
        "execution_sources": {"algorithm/observed_core.py": sources["algorithm/observed_core.py"]},
    }
    result = {"decision": "OBSERVED_RATE_AGE_TAIL_PERFORMANCE_PASS", "execution_checks_passed": True,
              "performance_gain_established": True, "registered_at_utc": "observed-dev-frozen"}
    generated = []

    def arrivals(rates, horizon, seed):
        generated.append((tuple(rates), horizon, seed))
        return [(float(seed % 100), 0, seed)]

    monkeypatch.setattr(confirmation, "poisson_arrivals", arrivals)
    return development, result, generated


def test_new_seed_population_preserves_causal_method_and_confirmation_scope(monkeypatch, tmp_path):
    development, result, generated = _development(monkeypatch, tmp_path)
    original = deepcopy(development)
    protocol = confirmation.build_protocol(development, result)
    assert development == original
    assert protocol["arrival_seeds"] == list(range(10001, 10013))
    assert not set(protocol["arrival_seeds"]) & set(range(1001, 1013))
    assert len(protocol["tapes"]) == len(generated) == 96
    assert len({tape["tape_id"] for tape in protocol["tapes"]}) == 96
    assert all(tape["arrivals"] == [[float(tape["seed"] % 100), 0, tape["seed"]]] for tape in protocol["tapes"])
    assert all(horizon == 600. and seed in range(10001, 10013) for _, horizon, seed in generated)
    for key in ("models", "capacities", "method", "clock", "information", "budget", "performance",
                "rate_estimator", "planning_samples", "planning_event_cap", "planning_seed_rule"):
        assert protocol[key] == development[key]
    assert protocol["evidence_type"] == "independent_confirmation"
    assert protocol["trajectory_accounting"] == {"primary_policies": 2, "primary_trajectories": 192,
        "reference_trajectories": 96, "total_executed_trajectories": 288}
    assert "the96" in protocol["primary"] and "Twelve held-out" in protocol["statistics"]
    assert "estimated only from causal" in protocol["limitations"]
    assert protocol["execution_sources"][confirmation.SOURCE] == "confirmation driver"
    assert confirmation.run_tape.__module__.endswith("aligned_hardware_observed_rate_age_tail_development")


def test_registration_requires_this_variant_matching_development_pass(monkeypatch, tmp_path):
    development, result, generated = _development(monkeypatch, tmp_path)
    for field, value in (("decision", "OBSERVED_RATE_AGE_TAIL_PERFORMANCE_HOLD"),
                         ("decision", "AGE_TAIL_CONSTRAINED_LANE_ROLLOUT_PERFORMANCE_PASS"),
                         ("execution_checks_passed", False), ("performance_gain_established", False),
                         ("registered_at_utc", "known-rate-development")):
        with pytest.raises(ValueError, match="development"):
            confirmation.build_protocol(development, {**result, field: value})
    assert generated == []


def test_frozen_confirmation_rejects_rate_rule_seed_tape_and_source_drift(monkeypatch, tmp_path):
    development, result, _ = _development(monkeypatch, tmp_path)
    protocol = {**confirmation.build_protocol(development, result), "registered_at_utc": "fresh-frozen"}
    confirmation.validate_registered_protocol(protocol, development, result)
    for field, value in (("arrival_seeds", list(range(1001, 1013))),
                         ("rate_estimator", {"rule": "true environment rates"}),
                         ("method", "retuned method"), ("planning_samples", 16)):
        with pytest.raises(ValueError, match="registered confirmation"):
            confirmation.validate_registered_protocol({**protocol, field: value}, development, result)
    changed = deepcopy(protocol)
    changed["tapes"][0]["arrivals"][0][0] += 1
    with pytest.raises(ValueError, match="registered confirmation"):
        confirmation.validate_registered_protocol(changed, development, result)
    (tmp_path / "algorithm/observed_core.py").write_text("changed estimator core")
    with pytest.raises(ValueError, match="dependency changed"):
        confirmation.build_protocol(development, result)


@pytest.mark.parametrize("integrity,performance,suffix", [(True, True, "PASS"), (True, False, "HOLD"),
                                                        (False, False, "REPLAY_FAIL")])
def test_unknown_variant_result_labels_and_held_out_seed_scope(monkeypatch, integrity, performance, suffix):
    monkeypatch.setattr(confirmation, "summarize", lambda rows, protocol: {
        "execution_checks_passed": integrity, "performance_gain_established": performance,
        "trajectories": 192, "independent_seed_replicates": 12})
    rows = [{"reference_work": {"selector_calls": 2, "option_probes": 18}} for _ in range(96)]
    protocol = {"candidate_method": "causal_count_rate_age_tail", "development_decision": confirmation.DEVELOPMENT_PASS,
                "development_registered_at_utc": "observed-dev-frozen"}
    result = confirmation.confirmation_summary(rows, protocol)
    assert result["decision"] == "OBSERVED_RATE_AGE_TAIL_CONFIRMATION_" + suffix
    assert result["candidate_method"] == "causal_count_rate_age_tail"
    assert result["evidence_type"] == "independent_confirmation"
    assert result["independent_seed_replicates"] == 12
    assert "Twelve held-out actual arrival seeds" in result["statistical_scope"]
    assert "conditional on the registered fixed planning stream" in result["statistical_scope"]
    assert result["development_decision"] == "OBSERVED_RATE_AGE_TAIL_PERFORMANCE_PASS"
    assert result["reference_trajectories"] == 96 and result["total_executed_trajectories"] == 288
