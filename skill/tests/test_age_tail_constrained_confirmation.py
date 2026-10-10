"""Confirmation must keep fresh scope, frozen methods and the old control."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

import algorithm.experiments.aligned_hardware_age_tail_constrained_confirmation as confirmation


def _development(monkeypatch, tmp_path):
    monkeypatch.setattr(confirmation, "ROOT", tmp_path)
    monkeypatch.setattr(confirmation, "DEVELOPMENT", tmp_path / "md/experiment_artifacts/development")
    paths = {"algorithm/core.py": "frozen core", confirmation.SOURCE: "confirmation driver"}
    for path, text in paths.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    models = [{"node": f"node{i}", "lanes": 4 if i == 1 else 2} for i in range(4)]
    development = {
        "registered_at_utc": "dev-frozen", "candidate_method": "FIFO_age_constraint",
        "models": models, "loads": [.8, .95], "arrival_seeds": [41, 42, 43],
        "tapes": [{"seed": 41, "arrivals": [[99., 0, 0]]}],
        "capacities": [{"node": m["node"], "lanes": m["lanes"], "load": load,
                        "arrival_rates": [load]} for m in models for load in [.8, .95]],
        "arrival_horizon": 600., "observation_end": 1200.,
        "execution_order": [[m["node"], m["lanes"]] for m in models],
        "performance": {"min_gain": .01, "group_secondary_tolerance": .05},
        "planning_samples": 8, "planning_event_cap": 128,
        "planning_seed_rule": "arrival-plan-v1:{node}:{load}:online", "clock": "frozen clock",
        "labels": ["maxweight", "arrival_rollout"], "method": "frozen age method",
        "information": "known rates and H", "budget": "frozen budget",
        "execution_sources": {"algorithm/core.py": paths["algorithm/core.py"]},
    }
    result = {"decision": confirmation.DEVELOPMENT_PASS, "execution_checks_passed": True,
              "performance_gain_established": True, "registered_at_utc": "dev-frozen"}
    monkeypatch.setattr(confirmation, "poisson_arrivals", lambda rates, horizon, seed: [(float(seed % 100), 0, seed)])
    return development, result


def test_fresh_population_and_model_rule_metadata_remain_frozen(monkeypatch, tmp_path):
    development, result = _development(monkeypatch, tmp_path)
    before = deepcopy(development)
    protocol = confirmation.build_protocol(development, result)
    assert development == before
    assert protocol["arrival_seeds"] == list(range(1001, 1013))
    assert len(protocol["tapes"]) == 96
    assert len({tape["tape_id"] for tape in protocol["tapes"]}) == 96
    assert all(tape["arrivals"] == [[float(tape["seed"] % 100), 0, tape["seed"]]] for tape in protocol["tapes"])
    for key in ("models", "capacities", "method", "clock", "budget", "information", "performance",
                "planning_samples", "planning_event_cap", "planning_seed_rule", "arrival_horizon", "observation_end"):
        assert protocol[key] == development[key]
    assert protocol["trajectory_accounting"] == {"primary_policies": 2, "primary_trajectories": 192,
        "reference_trajectories": 96, "total_executed_trajectories": 288}
    assert "the96" in protocol["primary"] and "Twelve held-out" in protocol["statistics"]
    assert "12 held-out" in protocol["limitations"] and protocol["evidence_type"] == "independent_confirmation"
    assert protocol["execution_sources"][confirmation.SOURCE] == "confirmation driver"


def test_confirmation_requires_the_matching_valid_development_pass(monkeypatch, tmp_path):
    development, result = _development(monkeypatch, tmp_path)
    for field, value in (("decision", "AGE_TAIL_CONSTRAINED_LANE_ROLLOUT_PERFORMANCE_HOLD"),
                         ("execution_checks_passed", False), ("performance_gain_established", False),
                         ("registered_at_utc", "another protocol")):
        changed = {**result, field: value}
        with pytest.raises(ValueError, match="development"):
            confirmation.build_protocol(development, changed)


def test_registered_fresh_scope_and_method_drift_are_rejected(monkeypatch, tmp_path):
    development, result = _development(monkeypatch, tmp_path)
    protocol = {**confirmation.build_protocol(development, result), "registered_at_utc": "fresh-frozen"}
    confirmation.validate_registered_protocol(protocol, development, result)
    for field, value in (("method", "changed method"), ("arrival_seeds", [41, 42, 43]),
                         ("planning_samples", 16), ("performance", {"min_gain": 0., "group_secondary_tolerance": 1.})):
        with pytest.raises(ValueError, match="registered confirmation"):
            confirmation.validate_registered_protocol({**protocol, field: value}, development, result)
    changed = deepcopy(protocol)
    changed["tapes"][0]["arrivals"][0][0] += 1
    with pytest.raises(ValueError, match="registered confirmation"):
        confirmation.validate_registered_protocol(changed, development, result)
    (tmp_path / "algorithm/core.py").write_text("changed core")
    with pytest.raises(ValueError, match="dependency changed"):
        confirmation.build_protocol(development, result)


def test_old_shadow_precedes_pair_and_is_normalized_and_separately_charged(monkeypatch):
    calls = []
    model = SimpleNamespace(node="node", lanes=2)
    tape = {"arrivals": [(1., 0, 0)]}
    protocol = {"arrival_horizon": 600., "observation_end": 1200.}

    def shadow(*args, **kwargs):
        calls.append("shadow")
        assert args == (model, tape["arrivals"], 600., 1200., 0, confirmation.select_single_lane_control)
        assert kwargs == {"policy": "maxweight", "dispatch_rule": "single"}
        return {"frames": [{"lane_ids": (0,), "queue": (1,)}], "jobs": []}, [
            SimpleNamespace(option_probes=9, candidate_evaluations=3),
            SimpleNamespace(option_probes=9, candidate_evaluations=4)]

    def pair(actual_model, actual_tape, actual_protocol, reference):
        calls.append("pair")
        assert actual_model is model and actual_tape is tape and actual_protocol is protocol
        assert reference["frames"][0] == {"lane_ids": [0], "queue": [1]}
        return {"baseline_reproduction_passed": True}, {"maxweight": {}, "arrival_rollout": {}}

    monkeypatch.setattr(confirmation, "simulate_lane_refill_trajectory", shadow)
    monkeypatch.setattr(confirmation, "run_tape", pair)
    row, policies = confirmation.run_confirmation_tape(model, tape, protocol)
    assert calls == ["shadow", "pair"] and len(policies) == 2
    assert row["reference_work"] == {"selector_calls": 2, "option_probes": 18, "candidate_evaluations": 7}


@pytest.mark.parametrize("integrity,performance,suffix", [(True, True, "PASS"), (True, False, "HOLD"),
                                                        (False, False, "REPLAY_FAIL")])
def test_confirmation_labels_seed_scope_and_reference_accounting(monkeypatch, integrity, performance, suffix):
    monkeypatch.setattr(confirmation, "summarize", lambda rows, protocol: {
        "execution_checks_passed": integrity, "performance_gain_established": performance,
        "trajectories": 192, "independent_seed_replicates": 12})
    rows = [{"reference_work": {"selector_calls": 2, "option_probes": 18}} for _ in range(96)]
    protocol = {"candidate_method": "age constraint", "development_decision": confirmation.DEVELOPMENT_PASS,
                "development_registered_at_utc": "dev-frozen"}
    result = confirmation.confirmation_summary(rows, protocol)
    assert result["decision"] == "AGE_TAIL_CONSTRAINED_LANE_ROLLOUT_CONFIRMATION_" + suffix
    assert result["reference_trajectories"] == 96 and result["total_executed_trajectories"] == 288
    assert result["reference_totals_work"] == {"selector_calls": 192, "option_probes": 1728}
    assert result["independent_seed_replicates"] == 12
    assert "conditional on the registered fixed planning stream" in result["statistical_scope"]
    assert result["evidence_type"] == "independent_confirmation"
