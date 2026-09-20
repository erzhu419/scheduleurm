from __future__ import annotations

from dataclasses import dataclass, replace
import itertools
import json

import pytest

from algorithm.canonical_count_space_bnb_frame_candidates import (
    select_canonical_count_space_bnb_frame,
)
from algorithm.experiments import (
    aligned_hardware_count_space_bnb_canonical_exact_compute_confirmation as confirmation,
)
from algorithm.experiments.aligned_frame_flow_decomposition import decompose_flow
from algorithm.experiments.aligned_hardware_frame_model import load_model


@dataclass(frozen=True)
class _Option:
    requirements: tuple[int, ...]
    duration: float


def _exhaustive_action(options, queues, lanes):
    option_count = len(options)
    best = None
    for ordered in itertools.product((None, *range(option_count)), repeat=lanes):
        indices = tuple(sorted(
            ordered,
            key=lambda index: option_count if index is None else index,
        ))
        service = tuple(
            sum(
                options[index].requirements[coordinate]
                for index in indices
                if index is not None
            )
            for coordinate in range(len(queues))
        )
        if any(done > waiting for done, waiting in zip(service, queues, strict=True)):
            continue
        duration = max((
            1.0,
            *(options[index].duration for index in indices if index is not None),
        ))
        score = sum(
            waiting * done
            for waiting, done in zip(queues, service, strict=True)
        ) / duration
        key = (duration, tuple(
            option_count if index is None else index for index in indices
        ))
        if best is None or score > best[0] or (score == best[0] and key < best[1]):
            best = (score, key, indices)
    return best[0], best[2]


def _stub_source():
    model = {
        "lanes": 2,
        "workloads": ["a", "b"],
        "options": [{"requirements": [1, 0]}, {"requirements": [0, 1]}],
    }
    return (
        {
            "score_only_result_path": "/tmp/stub-score-only.json",
            "score_only_experiment": confirmation.SCORE_ONLY_EXPERIMENT,
            "score_only_decision": confirmation.SOURCE_DECISION,
            "score_only_compute_confirmed": True,
            "score_only_evidence_disposition": (
                "retained for optimal-score and empirical-work claims; superseded "
                "by this v2 confirmation for canonical-policy claims"
            ),
            "prior_case_count": 4,
        },
        model,
        {"upstream.py": "frozen"},
    )


def test_small_exact_selection_action_matches_exhaustive_enumeration():
    options = (
        _Option((1, 0), 1.0),
        _Option((0, 1), 2.0),
        _Option((1, 1), 1.5),
    )
    queues = (3, 2)
    lanes = 3
    oracle_score, oracle_indices = _exhaustive_action(options, queues, lanes)
    selected = select_canonical_count_space_bnb_frame(
        options, queues, lanes, confirmation.LP_NODE_CAP
    )
    assert selected.exact
    assert selected.score_exact
    assert selected.canonical_exact
    assert selected.score == pytest.approx(oracle_score)
    assert selected.option_indices == oracle_indices
    assert selected.lp_calls < confirmation.LP_NODE_CAP


def test_small_trajectory_uses_one_exact_call_per_frame_and_reconstructs_fifo():
    model = replace(load_model(), lanes=8)
    arrivals = (
        (0.10, 0, 0),
        (0.20, 1, 0),
        (0.30, 2, 0),
        (0.40, 2, 1),
    )
    metrics, frames, flows = confirmation.simulate_cold_exact_frames(
        model, arrivals, arrival_horizon=2.0
    )
    decomposition = decompose_flow(model, arrivals, frames, 2.0)
    assert metrics["job_count"] == metrics["completed_jobs"] == len(flows) == 4
    assert metrics["selector_call_count"] == metrics["frame_count"]
    assert metrics["exact_selection_frame_count"] == metrics["frame_count"]
    assert metrics["score_exact_selection_frame_count"] == metrics["frame_count"]
    assert (
        metrics["canonical_exact_selection_frame_count"]
        == metrics["frame_count"]
    )
    assert metrics["valid_certificate_frame_count"] == metrics["frame_count"]
    assert len(metrics["actual_lp_calls_by_frame"]) == metrics["frame_count"]
    assert sum(metrics["actual_lp_calls_by_frame"]) == metrics["actual_work"][
        "lp_calls"
    ]
    assert metrics["maximum_actual_lp_calls"] <= confirmation.LP_NODE_CAP
    assert metrics["maximum_queue_recursion_violation"] == 0
    assert decomposition["mean_flow_s"] == pytest.approx(metrics["mean_flow_s"])


def test_nearest_rank_p95_and_decision_gates_are_frozen():
    values = [1] * 18 + [9, 10]
    assert confirmation._nearest_rank(values, 0.95) == 9
    passing = [{"case_gate_passed": True} for _ in range(4)]
    assert confirmation._decision(True, passing, True) == (
        "CANONICAL_EXACT_COMPUTE_CONFIRMATION_PASS"
    )
    assert confirmation._decision(True, passing, False) == (
        "CANONICAL_EXACT_COMPUTE_CONFIRMATION_FAIL_STOP"
    )
    assert confirmation._decision(False, passing, True) == "INVALID_EXECUTION"


def test_matrix_envelope_is_explicit_and_not_an_executed_baseline():
    _, model, _ = _stub_source()
    matrix = confirmation._matrix_envelope_constants(model)
    assert matrix["maximum_coefficients_per_lp"] == 2 * (1 + 2 + 2)
    assert "non-executed" in matrix["interpretation"]
    protocol = confirmation._protocol_constants(model, _stub_source()[0])
    assert protocol["matrix_work_accounting"]["gate"].startswith(
        "actual matrix sum"
    )
    assert protocol["lp_work_gates"]["mean_utilization"]["strict_upper"] == 0.05
    assert protocol["lp_work_gates"]["empirical_p95"]["upper_inclusive"] == 0.10
    assert protocol["selector_rule"]["required_score_exact_in_every_frame"]
    assert protocol["selector_rule"]["required_canonical_exact_in_every_frame"]
    assert protocol["static_roster_rng_seed_avoided"] == 10041
    assert "superseded" in protocol["source_contract"][
        "score_only_evidence_disposition"
    ]


def test_registration_does_no_model_capacity_tape_or_selector_work(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "confirmation.json"

    def forbidden(*args, **kwargs):
        raise AssertionError("registration performed execution work")

    monkeypatch.setattr(confirmation, "_source_contract", lambda path: _stub_source())
    monkeypatch.setattr(
        confirmation,
        "_current_fixed_sources",
        lambda upstream: {"upstream.py": "frozen", "confirmation.py": "frozen"},
    )
    monkeypatch.setattr(confirmation, "load_model", forbidden)
    monkeypatch.setattr(confirmation, "capacity_certificate", forbidden)
    monkeypatch.setattr(confirmation, "poisson_arrivals", forbidden)
    monkeypatch.setattr(confirmation, "select_canonical_count_space_bnb_frame", forbidden)

    protocol = confirmation.register(output, tmp_path / "score-only.json")
    result_path, protocol_path, fixed_path = confirmation._paths(output)
    assert not result_path.exists()
    assert protocol_path.exists() and fixed_path.exists()
    assert protocol["arrival_seeds"] == list(range(11041, 11049))
    assert protocol["loads"] == [0.8, 0.95]
    assert protocol["selector_rule"]["calls_per_frame"] == 1
    assert protocol["selector_rule"]["restarts"] == 0
    assert protocol["selector_rule"]["warm_starts"] == 0
    assert protocol["expected_work"]["ordered_action_enumerations"] == 0
    assert protocol["expected_work"]["count_vector_enumerations"] == 0


def test_run_rejects_protocol_change_before_model_or_tape_work(tmp_path, monkeypatch):
    output = tmp_path / "confirmation.json"
    monkeypatch.setattr(confirmation, "_source_contract", lambda path: _stub_source())
    monkeypatch.setattr(
        confirmation,
        "_current_fixed_sources",
        lambda upstream: {"upstream.py": "frozen", "confirmation.py": "frozen"},
    )
    confirmation.register(output, tmp_path / "score-only.json")
    _, protocol_path, _ = confirmation._paths(output)
    protocol = json.loads(protocol_path.read_text())
    protocol["arrival_horizon_s"] = 601.0
    protocol_path.write_text(json.dumps(protocol))

    def forbidden(*args, **kwargs):
        raise AssertionError("execution began before protocol validation")

    monkeypatch.setattr(confirmation, "load_model", forbidden)
    monkeypatch.setattr(confirmation, "poisson_arrivals", forbidden)
    with pytest.raises(ValueError, match="protocol changed"):
        confirmation.run_registered(output)
