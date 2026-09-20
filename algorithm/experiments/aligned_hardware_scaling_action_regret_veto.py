"""Post-result action-regret veto for the Cartesian-scaling certificate branch.

The frozen scaling result leaves R8 and R16 without a capacity-compatible
returned certificate at K4096.  This diagnostic reconstructs one explicit
feasible challenger for each reported alpha witness.  A challenger whose
support improvement already exceeds the smallest capacity margin proves that
no action-preserving upper-bound refinement can reverse that lane's quality
gate.  The cases were authored after inspecting the scaling result, so this is
a mechanism audit rather than independent preregistration.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from algorithm.budgeted_frame_candidates import select_budgeted_frame
from algorithm.experiments.aligned_hardware_frame_model import (
    capacity_certificate,
    load_model,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
SOURCE_RESULT = ARTIFACTS / (
    "aligned_hardware_cartesian_scaling_development_20260912.json"
)
SOURCE_PROTOCOL = ARTIFACTS / (
    "aligned_hardware_cartesian_scaling_development_20260912_protocol.json"
)
SOURCE_FIXED_SOURCES = ARTIFACTS / (
    "aligned_hardware_cartesian_scaling_development_20260912_fixed_sources.json"
)
DEFAULT_OUTPUT = ARTIFACTS / (
    "aligned_hardware_scaling_action_regret_veto_development_20260912.json"
)

EXPERIMENT = "aligned_hardware_scaling_action_regret_veto_v1"
SOURCE_EXPERIMENT = "aligned_hardware_cartesian_scaling_v1"
SOURCE_DECISION = "COMPUTE_SAVING_WITH_LOOSE_CERTIFICATE_STOP"
THIS_SOURCE = (
    "algorithm/experiments/aligned_hardware_scaling_action_regret_veto.py"
)
LOADS = (0.50, 0.80, 0.95)
SCORE_TOLERANCE = 1e-10

CASES = (
    {
        "case_id": "R8_K4096_alpha_witness",
        "lanes": 8,
        "budget": 4096,
        "base_queue": [5, 3, 5],
        "queue": [20, 12, 20],
        "source_option_indices": [7, 7, 7, 7, 7, 7, None, None],
        "challenger_option_indices": [7, 7, 7, 7, 7, 7, 5, 5],
    },
    {
        "case_id": "R16_K4096_alpha_witness",
        "lanes": 16,
        "budget": 4096,
        "base_queue": [5, 2, 4],
        "queue": [40, 16, 32],
        "source_option_indices": [
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            None, None, None,
        ],
        "challenger_option_indices": [
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            8, 8, 8, 8,
        ],
    },
)

DECISION_RULE = {
    "INVALID_EXECUTION": "one or more frozen reconstruction checks fail",
    "ACTION_REGRET_VETO_CERTIFICATE_ONLY_STOP": (
        "at least one fixed feasible challenger has normalized support gain "
        "greater than or equal to that lane's smallest load margin"
    ),
    "NO_ACTION_REGRET_VETO_ADVANCE_TO_CERTIFICATE_BOUND_TEST": (
        "every fixed challenger lower bound remains strictly below every "
        "capacity margin"
    ),
}

CLAIM_BOUNDARY = (
    "A post-result deterministic mechanism audit of two fixed synthetic-lane "
    "K4096 witnesses. It proves only a lower bound on selected-action regret; "
    "it is not an all-grid oracle, trajectory result, or independent confirmation."
)


def _json(value):
    return json.loads(json.dumps(value))


def _paths(output=DEFAULT_OUTPUT):
    output = Path(output)
    return (
        output,
        output.with_name(output.stem + "_protocol.json"),
        output.with_name(output.stem + "_fixed_sources.json"),
    )


def _write_new(path, payload, *, compact=False):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(
            payload,
            stream,
            indent=None if compact else 2,
            separators=(",", ":") if compact else None,
        )
        stream.write("\n")


def _load_source_bundle():
    source = json.loads(SOURCE_RESULT.read_text())
    protocol = json.loads(SOURCE_PROTOCOL.read_text())
    frozen = json.loads(SOURCE_FIXED_SOURCES.read_text())
    if source.get("protocol") != protocol:
        raise ValueError("scaling result does not embed its registered protocol")
    if (
        source.get("experiment") != SOURCE_EXPERIMENT
        or source.get("status") != "PASS"
        or source.get("scientific_decision") != SOURCE_DECISION
        or not source.get("checks")
        or not all(source["checks"].values())
        or any((ROOT / name).read_text() != text for name, text in frozen.items())
    ):
        raise ValueError("action-regret audit requires the frozen scaling bundle")
    return source, protocol, frozen


def _source_lane(source, lanes):
    return next(row for row in source["lane_summaries"] if row["lanes"] == lanes)


def _source_budget(source, lanes, budget):
    lane = _source_lane(source, lanes)
    return next(
        row for row in lane["budget_summaries"] if row["budget"] == budget
    )


def _source_capacities(source, lanes):
    lane = next(row for row in source["capacity_by_lane"] if row["lanes"] == lanes)
    return lane["certificates"]


def _case_source_contract(source, case):
    budget = _source_budget(source, case["lanes"], case["budget"])
    witness = budget["alpha_witness"]
    if (
        witness["base_queue"] != case["base_queue"]
        or witness["scaled_queue"] != case["queue"]
        or witness["option_indices"] != case["source_option_indices"]
    ):
        raise ValueError(f"fixed case no longer matches {case['case_id']}")
    return {
        "case": _json(case),
        "source_alpha_witness": _json(witness),
        "source_max_certified_normalized_gap": budget[
            "max_certified_normalized_gap"
        ],
        "source_capacity_certificates": _json(
            _source_capacities(source, case["lanes"])
        ),
    }


def _source_contract(source, source_protocol):
    return {
        "result": SOURCE_RESULT.name,
        "protocol": SOURCE_PROTOCOL.name,
        "experiment": source["experiment"],
        "status": source["status"],
        "scientific_decision": source["scientific_decision"],
        "model": _json(source_protocol["model"]),
        "cases": [_case_source_contract(source, case) for case in CASES],
    }


def _current_fixed_sources():
    _, _, upstream = _load_source_bundle()
    return {**upstream, THIS_SOURCE: (ROOT / THIS_SOURCE).read_text()}


def _protocol_constants(source, source_protocol):
    return {
        "experiment": EXPERIMENT,
        "role": "post-result deterministic action-regret mechanism audit",
        "post_result_diagnostic": True,
        "independent_preregistration": False,
        "source_contract": _source_contract(source, source_protocol),
        "loads": list(LOADS),
        "score_tolerance": SCORE_TOLERANCE,
        "expected_selector_calls": len(CASES),
        "expected_capacity_certificate_calls": len(CASES) * len(LOADS),
        "decision_rule": DECISION_RULE,
        "claim_boundary": CLAIM_BOUNDARY,
        "prohibited": (
            "full-grid reruns, full Cartesian action enumeration, oracle search, "
            "arrivals, trajectories, drains, RNG, server data, and paper edits"
        ),
    }


def register(output=DEFAULT_OUTPUT):
    """Freeze the two post-result cases without calling the selector."""

    output, protocol_path, fixed_sources_path = _paths(output)
    if any(path.exists() for path in (output, protocol_path, fixed_sources_path)):
        raise ValueError("action-regret audit artifacts already exist")
    source, source_protocol, _ = _load_source_bundle()
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        **_protocol_constants(source, source_protocol),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_new(protocol_path, protocol, compact=True)
    _write_new(fixed_sources_path, _current_fixed_sources())
    return protocol


def _action_row(model, queue, option_indices):
    indices = tuple(option_indices)
    service = tuple(
        sum(
            model.options[index].requirements[coordinate]
            for index in indices
            if index is not None
        )
        for coordinate in range(len(queue))
    )
    duration = max((
        1.0,
        *(
            model.options[index].duration
            for index in indices
            if index is not None
        ),
    ))
    numerator = sum(
        waiting * completed
        for waiting, completed in zip(queue, service, strict=True)
    )
    return {
        "option_indices": list(indices),
        "service": list(service),
        "duration": duration,
        "support_numerator": numerator,
        "score": numerator / duration,
        "feasible": all(
            completed <= waiting
            for completed, waiting in zip(service, queue, strict=True)
        ),
    }


def _close(left, right):
    return math.isclose(
        float(left), float(right), rel_tol=0.0, abs_tol=SCORE_TOLERANCE
    )


def _decision(execution_passed, cases):
    if not execution_passed:
        return "INVALID_EXECUTION"
    if any(row["action_regret_veto"] for row in cases):
        return "ACTION_REGRET_VETO_CERTIFICATE_ONLY_STOP"
    return "NO_ACTION_REGRET_VETO_ADVANCE_TO_CERTIFICATE_BOUND_TEST"


def _case_result(model, contract):
    case = contract["case"]
    queue = tuple(case["queue"])
    lanes = case["lanes"]
    selection = select_budgeted_frame(
        model.options, queue, lanes, case["budget"]
    )
    selected = _action_row(model, queue, case["source_option_indices"])
    challenger = _action_row(model, queue, case["challenger_option_indices"])

    capacities = []
    for load in LOADS:
        certificate = capacity_certificate(replace(model, lanes=lanes), load)
        capacities.append({
            "load": load,
            "effective_slack_eta": certificate["effective_slack_eta"],
            "capacity_projection_passed": certificate[
                "capacity_projection_passed"
            ],
        })
    source_capacities = contract["source_capacity_certificates"]
    capacities_reproduce = all(
        row["load"] == expected["load"]
        and _close(
            row["effective_slack_eta"], expected["effective_slack_eta"]
        )
        and row["capacity_projection_passed"]
        == expected["capacity_projection_passed"]
        for row, expected in zip(capacities, source_capacities, strict=True)
    )

    queue_norm = sum(queue)
    regret_lower_bound = challenger["score"] - selected["score"]
    normalized_lower_bound = regret_lower_bound / queue_norm
    minimum_eta = min(row["effective_slack_eta"] for row in capacities)
    residual_margin_upper_bound = minimum_eta - normalized_lower_bound
    veto = residual_margin_upper_bound <= SCORE_TOLERANCE
    witness = contract["source_alpha_witness"]
    selection_reproduces = (
        list(selection.option_indices) == witness["option_indices"]
        and selection.attempts == witness["attempts"]
        and selection.exact == witness["exact"]
        and _close(selection.score, witness["score"])
        and _close(selection.upper_bound, witness["upper_bound"])
        and _close(selection.gap, witness["gap"])
        and _close(selection.epsilon, witness["value"])
    )
    selected_reconstructs = (
        list(selection.service) == selected["service"]
        and _close(selection.duration, selected["duration"])
        and _close(selection.score, selected["score"])
    )
    return {
        "case_id": case["case_id"],
        "lanes": lanes,
        "budget": case["budget"],
        "base_queue": case["base_queue"],
        "queue": list(queue),
        "queue_l1": queue_norm,
        "source_selection": {
            "option_indices": list(selection.option_indices),
            "service": list(selection.service),
            "duration": selection.duration,
            "score": selection.score,
            "upper_bound": selection.upper_bound,
            "gap": selection.gap,
            "epsilon": selection.epsilon,
            "attempts": selection.attempts,
            "exact": selection.exact,
        },
        "selected_action_reconstruction": selected,
        "challenger": challenger,
        "capacity_certificates": capacities,
        "minimum_effective_slack_eta": minimum_eta,
        "challenger_support_gain": regret_lower_bound,
        "normalized_action_regret_lower_bound": normalized_lower_bound,
        "best_possible_action_preserving_residual_margin_upper_bound": (
            residual_margin_upper_bound
        ),
        "action_regret_veto": veto,
        "checks": {
            "fresh_selector_reproduces_source_alpha_witness": selection_reproduces,
            "selected_action_independently_reconstructs": selected_reconstructs,
            "challenger_has_exactly_one_choice_per_lane": (
                len(challenger["option_indices"]) == lanes
            ),
            "challenger_is_feasible": challenger["feasible"],
            "challenger_strictly_improves_support": (
                regret_lower_bound > SCORE_TOLERANCE
            ),
            "capacity_certificates_reproduce_source": capacities_reproduce,
            "normalized_lower_bound_arithmetic_closes": _close(
                normalized_lower_bound,
                (challenger["score"] - selection.score) / queue_norm,
            ),
        },
    }


def run_registered(output=DEFAULT_OUTPUT):
    output, protocol_path, fixed_sources_path = _paths(output)
    if output.exists():
        raise ValueError("action-regret audit result already exists")
    protocol = json.loads(protocol_path.read_text())
    frozen_sources = json.loads(fixed_sources_path.read_text())
    source, source_protocol, _ = _load_source_bundle()
    expected = _protocol_constants(source, source_protocol)
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("registered action-regret audit protocol changed")
    if frozen_sources != _current_fixed_sources():
        raise ValueError("registered action-regret audit sources changed")

    model = load_model()
    if _json(model.snapshot()) != protocol["source_contract"]["model"]:
        raise ValueError("current aligned model differs from the registered source")

    cases = [
        _case_result(model, contract)
        for contract in protocol["source_contract"]["cases"]
    ]
    checks = {
        "two_fixed_post_result_cases_executed_once": (
            len(cases) == len(CASES) == 2
            and [row["case_id"] for row in cases]
            == [case["case_id"] for case in CASES]
        ),
        "all_case_reconstruction_checks_pass": all(
            all(row["checks"].values()) for row in cases
        ),
        "registered_sources_unchanged": (
            frozen_sources == _current_fixed_sources()
        ),
        "no_full_grid_or_oracle_trajectory_work": True,
    }
    execution_passed = all(checks.values())
    decision = _decision(execution_passed, cases)
    result = {
        "experiment": EXPERIMENT,
        "status": "PASS" if execution_passed else "FAIL",
        "checks": checks,
        "protocol": protocol,
        "post_result_diagnostic": True,
        "independent_preregistration": False,
        "cases": cases,
        "veto_case_count": sum(row["action_regret_veto"] for row in cases),
        "work_accounting": {
            "selector_calls": len(cases),
            "capacity_certificate_calls": len(cases) * len(LOADS),
            "fixed_challenger_reconstructions": len(cases),
            "full_grid_states": 0,
            "full_action_or_oracle_enumerations": 0,
            "arrival_tapes": 0,
            "trajectory_frames": 0,
            "rng_initializations": 0,
            "random_draws": 0,
        },
        "scientific_decision": decision,
        "development_advance": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "limitations": [
            "The two challengers were authored after inspecting the scaling result.",
            "A feasible challenger proves a regret lower bound, not global optimality.",
            "The replicated R8 and R16 lanes remain synthetic and omit added hardware interference.",
        ],
        "next_step": (
            "close action-preserving certificate-only tightening at the frozen "
            "K4096 endpoints and diagnose scalable candidate discovery"
            if decision == "ACTION_REGRET_VETO_CERTIFICATE_ONLY_STOP"
            else "freeze a separate certificate-bound test before running it"
        ),
    }
    if not execution_passed:
        raise ValueError("action-regret veto audit checks failed")
    _write_new(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run-registered", action="store_true")
    args = parser.parse_args()
    payload = (
        register(args.output)
        if args.register
        else run_registered(args.output)
    )
    print(json.dumps({
        "experiment": payload["experiment"],
        "status": payload.get("status", "REGISTERED"),
        "scientific_decision": payload.get("scientific_decision"),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
