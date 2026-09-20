from __future__ import annotations

import json
from pathlib import Path

import pytest

from algorithm.experiments import aligned_hardware_joint95_canonical_selector_screen as screen
from algorithm.experiments.aligned_hardware_frame_model import load_model


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _capacity_copy(tmp_path: Path) -> tuple[Path, Path]:
    source = json.loads(screen.DEFAULT_CAPACITY_REGISTRATION.read_text(encoding="utf-8"))
    results = (tmp_path / "joint-results").resolve()
    results.mkdir(parents=True)
    source["source_paths"]["joint_results_directory"] = str(results)
    source["source_paths"]["holdout_release"] = str(results / "holdout_release.json")
    path = tmp_path / "capacity-protocol.json"
    _write(path, source)
    return path, results


def _register_tmp(tmp_path: Path):
    capacity, _ = _capacity_copy(tmp_path)
    registration = tmp_path / "screen-protocol.json"
    roster = tmp_path / "screen-roster.json"
    fixed = tmp_path / "screen-fixed.json"
    future_capacity = tmp_path / "future-capacity.json"
    protocol = screen.register(
        registration,
        roster,
        fixed,
        capacity_registration_path=capacity,
        capacity_result_path=future_capacity,
    )
    return protocol, registration, roster, fixed, future_capacity


def test_formal_canonical_confirmation_is_current_and_accepted():
    contract, fixed = screen._canonical_confirmation_contract(
        screen.DEFAULT_CANONICAL_CONFIRMATION
    )
    assert contract["status"] == "PASS"
    assert contract["scientific_decision"] == screen.CANONICAL_SOURCE_DECISION
    assert contract["canonical_exact_compute_confirmed"] is True
    assert all(contract["checks"].values())
    assert contract["aggregate_gate"]["aggregate_gate_passed"] is True
    assert "algorithm/canonical_count_space_bnb_frame_candidates.py" in fixed


def test_seed_13041_roster_is_unique_and_disjoint_from_every_old_roster():
    count_payload = json.loads(screen.DEFAULT_COUNT_ROSTER.read_text(encoding="utf-8"))
    count_queues = screen._validated_count_roster(count_payload)
    static_payload = json.loads(
        screen.DEFAULT_LEGACY_STATIC_ROSTER.read_text(encoding="utf-8")
    )
    static_queues = screen._validated_legacy_static_roster(
        static_payload, count_queues
    )
    first = screen.generate_roster(count_queues, static_queues)
    second = screen.generate_roster(count_queues, static_queues)
    queues = {tuple(row) for row in first["queues"]}
    assert first == second
    assert first["seed"] == 13041
    assert first["candidate_draws"] == 545
    assert first["rng_integer_draws"] == 1635
    assert first["queues"][:3] == [[26, 6, 13], [47, 19, 28], [21, 2, 6]]
    assert len(queues) == 512
    assert not queues & screen._old_base_grid()
    assert not queues & set(count_queues)
    assert not queues & set(static_queues)
    assert (0, 0, 0) not in queues


def test_registration_reads_no_joint_or_capacity_outcome_and_invokes_no_selector(
    tmp_path, monkeypatch
):
    capacity, joint_results = _capacity_copy(tmp_path)
    future_capacity = tmp_path / "future-capacity.json"
    (joint_results / "poison.json").write_text("not-json", encoding="utf-8")
    future_capacity.write_text("not-json", encoding="utf-8")
    original_read = screen._read_object
    reads = []

    def guarded_read(path):
        resolved = Path(path).resolve()
        reads.append(resolved)
        if resolved == future_capacity.resolve() or joint_results in resolved.parents:
            raise AssertionError("registration read future physical evidence")
        return original_read(path)

    def forbidden(*args, **kwargs):
        raise AssertionError("registration invoked scientific execution")

    monkeypatch.setattr(screen, "_read_object", guarded_read)
    monkeypatch.setattr(screen, "select_canonical_count_space_bnb_frame", forbidden)
    monkeypatch.setattr(screen.legacy, "_runtime_context", forbidden)
    monkeypatch.setattr(screen.legacy, "_physical_capacity_rows", forbidden)
    registration = tmp_path / "registered.json"
    roster = tmp_path / "roster.json"
    fixed = tmp_path / "fixed.json"
    protocol = screen.register(
        registration,
        roster,
        fixed,
        capacity_registration_path=capacity,
        capacity_result_path=future_capacity,
    )
    scope = protocol["registration_scope"]
    assert scope["joint_results_directory_read"] is False
    assert scope["joint_visit_values_read"] is False
    assert scope["capacity_closure_result_read"] is False
    assert scope["joint_margin_computed"] is False
    assert scope["capacity_computed"] is False
    assert scope["selector_invoked"] is False
    assert scope["exact_reference_invoked"] is False
    assert scope["roster_rng_initializations"] == 1
    assert scope["other_rng_draws"] == 0
    assert future_capacity.resolve() not in reads
    assert all(joint_results not in path.parents for path in reads)
    assert registration.is_file() and roster.is_file() and fixed.is_file()


def test_run_refuses_incomplete_joint_collection_before_capacity_or_selector(
    tmp_path, monkeypatch
):
    _, registration, roster, fixed, _ = _register_tmp(tmp_path)
    output = tmp_path / "outcome.json"

    def forbidden(*args, **kwargs):
        raise AssertionError("incomplete joint evidence reached scientific work")

    monkeypatch.setattr(screen, "select_canonical_count_space_bnb_frame", forbidden)
    monkeypatch.setattr(screen.legacy, "_physical_capacity_rows", forbidden)
    with pytest.raises(screen.CanonicalStaticScreenRefusal, match="incomplete or invalid"):
        screen.run_registered(registration, roster, fixed, output)
    assert not output.exists()


def test_one_state_screen_uses_canonical_exact_reference_and_native_work(
    monkeypatch,
):
    monkeypatch.setattr(screen, "ROSTER_SIZE", 1)
    model = load_model()
    capacity = [{
        "load": load,
        "load_normalization": "fraction_of_final_joint_envelope_common_capacity_boundary",
        "arrival_rates": [0.1, 0.1, 0.1],
        "effective_slack_eta": 100.0,
        "capacity_projection_passed": True,
    } for load in screen.LOADS]
    row, checks = screen._screen_lane(model, [(26, 6, 13)], 8, capacity)
    assert all(checks.values())
    assert row["state_count"] == 1
    assert row["valid_canonical_reference_count"] == 1
    assert row["regret_bounded_by_epsilon_count"] == 1
    assert row["support_gate_passed"] is True
    assert row["candidate_actual_work"]["lp_calls"] <= 36
    assert row["cold_canonical_reference_actual_work"]["lp_calls"] <= 4096
    assert all(
        witness["cold_canonical_reference"]["score_exact"]
        and witness["cold_canonical_reference"]["canonical_exact"]
        for witness in (row["epsilon_witness"], row["support_regret_witness"])
    )


@pytest.mark.parametrize(
    ("execution", "support", "compute", "expected"),
    [
        (False, True, True, "INVALID_EXECUTION"),
        (True, False, True, "CANONICAL_PHYSICAL_SELECTOR_SUPPORT_FAIL_STOP"),
        (True, True, False, "CANONICAL_PHYSICAL_SELECTOR_COMPUTE_FAIL_STOP"),
        (True, True, True, screen.PASS_DECISION),
    ],
)
def test_frozen_decision_branches(execution, support, compute, expected):
    lanes = [
        {"support_gate_passed": support, "actual_compute_gate_passed": compute},
        {"support_gate_passed": support, "actual_compute_gate_passed": compute},
    ]
    assert screen._decision(execution, lanes, compute) == expected


def test_protocol_retains_old_evidence_but_supersedes_trajectory_authority(tmp_path):
    protocol, *_ = _register_tmp(tmp_path)
    disposition = protocol["legacy_static_screen_disposition"]
    assert "support-score" in disposition["evidence_retained"]
    assert disposition["superseded_scope"].startswith("trajectory authorization only")
    assert disposition["artifact_erased_or_rewritten"] is False
    assert protocol["selector_contract"]["fixed_budget_by_lane"] == {
        "8": 36,
        "16": 72,
    }
    assert protocol["selector_contract"]["cold_canonical_reference_budget"] == 4096
    assert protocol["physical_load_contract"]["loads"] == [0.5, 0.8, 0.95]
    assert protocol["expected_work"]["arrival_tapes"] == 0
    assert "flow or trajectory result" in protocol["claim_boundary"]
