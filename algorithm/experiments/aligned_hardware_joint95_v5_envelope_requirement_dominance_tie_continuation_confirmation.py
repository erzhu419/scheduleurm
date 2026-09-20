"""Independent confirmation of the fixed requirement-dominance trajectory.

Registration binds the audited development PASS and freezes fresh arrival seeds
12101--12120 before any random draw, selector call, or trajectory execution.
Execution prepares the fixed nine-to-eight dominance quotient once, then reuses
the development worker unchanged: the candidate searches the eight-option
quotient with score-exact tie continuation, every candidate frame receives a
fresh full nine-option cold K4096 shadow, and the executed reference is the full
nine-option cold K4096 policy.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from algorithm.experiments import (
    aligned_hardware_joint95_v5_envelope_requirement_dominance_tie_continuation_development
    as development,
)


confirmation = development.confirmation
raw = development.raw
prepare_canonical_count_space_bnb_requirement_dominance_catalogue = (
    development.prepare_canonical_count_space_bnb_requirement_dominance_catalogue
)
ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
STEM = (
    "aligned_hardware_joint95_v5_envelope_"
    "requirement_dominance_tie_continuation_confirmation_20260920"
)
DEFAULT_REGISTRATION = ARTIFACTS / f"{STEM}_protocol.json"
DEFAULT_FIXED_SOURCES = ARTIFACTS / f"{STEM}_fixed_sources.json"
DEFAULT_RESULT = ARTIFACTS / f"{STEM}.json"
DEFAULT_DEVELOPMENT_REGISTRATION = development.DEFAULT_REGISTRATION
DEFAULT_DEVELOPMENT_FIXED_SOURCES = development.DEFAULT_FIXED_SOURCES
DEFAULT_DEVELOPMENT_RESULT = development.DEFAULT_RESULT

EXPERIMENT = (
    "aligned_hardware_joint95_v5_envelope_"
    "requirement_dominance_tie_continuation_confirmation_v1"
)
GATE = (
    "aligned_hardware_joint95_v5_envelope_"
    "requirement_dominance_tie_continuation_confirmation_runner"
)
REGISTRATION_STATUS = (
    "REGISTERED_AWAITING_INDEPENDENT_REQUIREMENT_DOMINANCE_"
    "TIE_CONTINUATION_TRAJECTORY"
)
PASS_DECISION = "INDEPENDENT_REQUIREMENT_DOMINANCE_QUOTIENT_TRAJECTORY_PASS"
INVALID_DECISION = (
    "INVALID_REQUIREMENT_DOMINANCE_TIE_CONTINUATION_CONFIRMATION_EXECUTION"
)
SUPPORT_FAIL_DECISION = (
    "REQUIREMENT_DOMINANCE_TIE_CONTINUATION_CONFIRMATION_SUPPORT_FAIL_STOP"
)
ACTION_FAIL_DECISION = (
    "REQUIREMENT_DOMINANCE_TIE_CONTINUATION_CONFIRMATION_"
    "ACTION_OR_REGRET_FAIL_STOP"
)
COMPUTE_FAIL_DECISION = (
    "REQUIREMENT_DOMINANCE_TIE_CONTINUATION_CONFIRMATION_COMPUTE_FAIL_STOP"
)
FLOW_FAIL_DECISION = (
    "REQUIREMENT_DOMINANCE_TIE_CONTINUATION_CONFIRMATION_"
    "FLOW_NONINFERIORITY_FAIL_STOP"
)

LANES = development.LANES
LOADS = development.LOADS
ARRIVAL_SEEDS = tuple(range(12101, 12121))
PREVIOUSLY_USED_ARRIVAL_SEEDS = tuple(
    sorted(
        {
            *confirmation.PREVIOUSLY_USED_ARRIVAL_SEEDS,
            *confirmation.ARRIVAL_SEEDS,
        }
    )
)
ARRIVAL_HORIZON_S = development.ARRIVAL_HORIZON_S
PRIMARY_BUDGETS = development.PRIMARY_BUDGETS
CONTINUATION_BUDGETS = development.CONTINUATION_BUDGETS
REFERENCE_BUDGET = development.REFERENCE_BUDGET
WORK_FIELDS = development.WORK_FIELDS
WORK_GATE_FIELDS = development.WORK_GATE_FIELDS
EXECUTION_WORKERS = development.EXECUTION_WORKERS
ORIGINAL_OPTION_COUNT = development.ORIGINAL_OPTION_COUNT
QUOTIENT_OPTION_COUNT = development.QUOTIENT_OPTION_COUNT
THIS_SOURCE = (
    "algorithm/experiments/"
    "aligned_hardware_joint95_v5_envelope_"
    "requirement_dominance_tie_continuation_confirmation.py"
)
TEST_SOURCE = (
    "skill/tests/"
    "test_aligned_hardware_joint95_v5_envelope_"
    "requirement_dominance_tie_continuation_confirmation.py"
)
HISTORICAL_SOURCE_PREFIX = confirmation.HISTORICAL_SOURCE_PREFIX


class RequirementDominanceConfirmationRefusal(RuntimeError):
    pass


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RequirementDominanceConfirmationRefusal(
            f"expected JSON object: {path}"
        )
    return payload


def _write_new(path: Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=None if compact else 2, sort_keys=True)
        handle.write("\n")


def _source_paths(
    development_registration: Path = DEFAULT_DEVELOPMENT_REGISTRATION,
    development_fixed_sources: Path = DEFAULT_DEVELOPMENT_FIXED_SOURCES,
    development_result: Path = DEFAULT_DEVELOPMENT_RESULT,
) -> dict[str, str]:
    return {
        "development_registration": str(Path(development_registration).resolve()),
        "development_fixed_sources": str(Path(development_fixed_sources).resolve()),
        "development_result": str(Path(development_result).resolve()),
    }


def _current_fixed_sources(
    development_fixed_sources: Path = DEFAULT_DEVELOPMENT_FIXED_SOURCES,
) -> dict[str, str]:
    rows = _read_object(Path(development_fixed_sources))
    if THIS_SOURCE in rows:
        raise RequirementDominanceConfirmationRefusal(
            "confirmation source already exists in predecessor bundle"
        )
    rows[THIS_SOURCE] = (ROOT / THIS_SOURCE).read_text(encoding="utf-8")
    return dict(sorted(rows.items()))


def _validate_seed_roster() -> None:
    if (
        tuple(development.ARRIVAL_SEEDS) != tuple(range(12061, 12101))
        or tuple(development.NEXT_CONFIRMATION_SEEDS) != ARRIVAL_SEEDS
        or len(PREVIOUSLY_USED_ARRIVAL_SEEDS) != 173
        or ARRIVAL_SEEDS != tuple(range(12101, 12121))
        or len(set(ARRIVAL_SEEDS)) != 20
        or set(PREVIOUSLY_USED_ARRIVAL_SEEDS).intersection(ARRIVAL_SEEDS)
    ):
        raise RequirementDominanceConfirmationRefusal(
            "independent arrival-seed roster is not the frozen disjoint successor"
        )


def _validated_development_bundle(
    paths: Mapping[str, str],
    quotient_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate the exact development PASS without preparing a second quotient."""

    if set(paths) != {
        "development_registration",
        "development_fixed_sources",
        "development_result",
    }:
        raise RequirementDominanceConfirmationRefusal("development paths changed")
    protocol = _read_object(Path(paths["development_registration"]))
    frozen = _read_object(Path(paths["development_fixed_sources"]))
    result = _read_object(Path(paths["development_result"]))
    source_paths = protocol.get("source_paths") or {}
    try:
        predecessor_contract, predecessor_protocol, capacity, eta = (
            development._validated_confirmation_bundle(source_paths)
        )
        expected = development._registration_body(
            source_paths,
            predecessor_contract,
            predecessor_protocol,
            capacity,
            eta,
            quotient_contract,
        )
        development._validate_protocol(protocol, expected)
        expected_frozen = development._current_fixed_sources(
            Path(source_paths["confirmation_fixed_sources"])
        )
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        raise RequirementDominanceConfirmationRefusal(
            f"development lineage is invalid: {exc}"
        ) from exc

    normal_keys = sorted(
        key for key in frozen if not key.startswith(HISTORICAL_SOURCE_PREFIX)
    )
    drift = [
        key
        for key in normal_keys
        if not (ROOT / key).is_file()
        or (ROOT / key).read_text(encoding="utf-8") != frozen[key]
    ]
    if (
        len(frozen) != 65
        or len(normal_keys) != 35
        or frozen != expected_frozen
        or drift
        or development.THIS_SOURCE not in frozen
        or development.REDUCER_SOURCE not in frozen
        or development.QUOTIENT_SOURCE not in frozen
    ):
        raise RequirementDominanceConfirmationRefusal(
            "development 65-source bundle or current normal sources changed"
        )

    checks = result.get("checks") or {}
    cases = result.get("case_summaries") or ()
    runs = result.get("runs") or ()
    work_gates = result.get("work_gates") or {}
    expected_run_keys = [
        (lanes, load, seed)
        for lanes in LANES
        for load in LOADS
        for seed in development.ARRIVAL_SEEDS
    ]
    observed_run_keys = [
        (int(row["lanes"]), float(row["load"]), int(row["seed"]))
        for row in runs
    ]
    lane_rows = {
        int(row["lanes"]): row for row in work_gates.get("by_lane") or ()
    }
    aggregate_fields = (
        (work_gates.get("aggregate") or {}).get("comparison", {}).get("fields", {})
    )
    if (
        result.get("protocol") != protocol
        or result.get("gate") != development.GATE
        or result.get("experiment") != development.EXPERIMENT
        or result.get("status") != "PASS"
        or result.get("scientific_decision") != development.PASS_DECISION
        or result.get("development_replay_passed") is not True
        or result.get("new_seed_independent_confirmation_authorized") is not True
        or result.get("authorized_confirmation_seeds") != list(ARRIVAL_SEEDS)
        or result.get("independent_confirmation_eligible") is not False
        or result.get("trajectory_or_flow_claim_authorized") is not False
        or result.get("paper_claim_authorized") is not False
        or result.get("trajectory_runs") != 480
        or result.get("arrival_generations") != 240
        or result.get("shadow_trajectory_runs") != 0
        or result.get("server_pulls") != 0
        or result.get("tie_continuation_trigger_count") != 0
        or result.get("tie_continuation_final_exact_count") != 0
        or result.get(
            "all_triggered_development_states_reached_canonical_exact"
        )
        is not True
        or result.get("catalogue_preprocessing_work")
        != protocol.get("quotient_contract", {}).get("reduction")
        or not checks
        or not all(checks.values())
        or len(cases) != 6
        or not all(
            row.get("execution_integrity_passed") is True
            and row.get("trajectory_support_gate_passed") is True
            and row.get("same_state_action_and_regret_gate_passed") is True
            and row.get("paired_mean_flow_noninferiority_passed") is True
            and row.get("case_compute_gate_passed") is True
            and row.get("case_gate_passed") is True
            and row.get("seeds") == list(development.ARRIVAL_SEEDS)
            and row.get("candidate_bnb_option_count") == QUOTIENT_OPTION_COUNT
            and row.get("shadow_and_reference_option_count")
            == ORIGINAL_OPTION_COUNT
            and all(
                row["charged_work_comparison"]["fields"][field]["no_greater"]
                is True
                for field in WORK_GATE_FIELDS
            )
            for row in cases
        )
        or observed_run_keys != expected_run_keys
        or work_gates.get("gate_passed") is not True
        or work_gates.get("uniform_case_no_regression_gate_passed") is not True
        or set(lane_rows) != {8, 16}
        or not all(row.get("gate_passed") is True for row in lane_rows.values())
        or not (work_gates.get("aggregate") or {}).get("gate_passed")
        or not all(
            lane_rows[lanes]["comparison"]["fields"][field]["strictly_lower"]
            is True
            for lanes in LANES
            for field in WORK_GATE_FIELDS
        )
        or not all(
            aggregate_fields[field]["strictly_lower"] is True
            for field in WORK_GATE_FIELDS
        )
    ):
        raise RequirementDominanceConfirmationRefusal(
            "development is not the exact audited dominance-quotient PASS"
        )

    contract = {
        "experiment": result["experiment"],
        "gate": result["gate"],
        "status": result["status"],
        "scientific_decision": result["scientific_decision"],
        "arrival_seeds": list(development.ARRIVAL_SEEDS),
        "authorized_confirmation_seeds": list(ARRIVAL_SEEDS),
        "arrival_generations": result["arrival_generations"],
        "trajectory_runs": result["trajectory_runs"],
        "tie_continuation_trigger_count": result["tie_continuation_trigger_count"],
        "tie_continuation_final_exact_count": result[
            "tie_continuation_final_exact_count"
        ],
        "all_six_case_gates_passed": True,
        "both_lane_compute_gates_passed": True,
        "aggregate_compute_gate_passed": True,
        "quotient_contract": _json(protocol["quotient_contract"]),
        "fixed_source_keys": sorted(frozen),
        "fixed_source_count": len(frozen),
        "historical_source_count": len(frozen) - len(normal_keys),
        "current_normal_source_keys": normal_keys,
        "current_normal_source_drift": [],
        "independent_confirmation_authorized": True,
    }
    capacity = protocol.get("physical_envelope_capacity_normalized_loads")
    eta = protocol.get("lane_minimum_effective_slack_eta")
    if not isinstance(capacity, dict) or not isinstance(eta, dict):
        raise RequirementDominanceConfirmationRefusal(
            "development envelope is missing"
        )
    return contract, protocol, _json(capacity), _json(eta)


METHOD_CONTRACT = _json(development.METHOD_CONTRACT)

TRAJECTORY_CONTRACT = {
    "lanes": list(LANES),
    "loads": list(LOADS),
    "arrival_seeds": list(ARRIVAL_SEEDS),
    "previously_used_arrival_seeds": list(PREVIOUSLY_USED_ARRIVAL_SEEDS),
    "arrival_horizon_s": ARRIVAL_HORIZON_S,
    "case_seed_tapes": len(LANES) * len(LOADS) * len(ARRIVAL_SEEDS),
    "executed_policies": ["candidate", "cold_k4096_reference"],
    "trajectory_runs": 2 * len(LANES) * len(LOADS) * len(ARRIVAL_SEEDS),
    "candidate_method": METHOD_CONTRACT,
    "candidate_state_shadow": (
        "one fresh full nine-option cold K4096 call on every candidate "
        "pre-dispatch state; audit only and never read by the candidate"
    ),
    "shadow_trajectory_runs": 0,
    "common_tape_rule": (
        "one immutable tape per R/load/seed replayed by both executed arms"
    ),
    "dispatch": "global synchronous complete-job frames with FIFO within class",
    "online_information": (
        "the candidate reads only the released current queue, model, lanes and "
        "registered budgets"
    ),
    "stopping_rule": "stop arrivals at 600 seconds and fully drain both arms",
    "reference_budget": REFERENCE_BUDGET,
    "paired_endpoint": "candidate/reference arithmetic mean-flow ratio",
    "noninferiority_margin": raw.NONINFERIORITY_MARGIN,
    "family_alpha": raw.FAMILY_ALPHA,
    "primary_family_size": raw.PRIMARY_FAMILY_SIZE,
}

REGISTRATION_SCOPE = {
    "development_pass_read": True,
    "development_protocol_and_fixed_sources_read": True,
    "independent_confirmation": True,
    "catalogue_reduction_and_lift_preparations": 1,
    "rng_initializations": 0,
    "rng_draws": 0,
    "arrival_generations": 0,
    "selector_calls": 0,
    "trajectory_runs": 0,
    "server_pulls": 0,
    "paper_or_readme_edits": 0,
    "test_source_in_fixed_bundle": False,
    "test_source_exclusion_reason": (
        "verification-only tests are not imported or executed by the formal run; "
        "the executable confirmation runner is frozen as source 66"
    ),
}

CLAIM_BOUNDARY = (
    "A PASS is independent confirmation of the fixed eight-option requirement-"
    "dominance quotient with score-exact tie continuation only for this "
    "preregistered nine-option model, R8/R16, 0.5/0.8/0.95 normalized loads, "
    "12101--12120 synthetic Poisson tapes, 600-second arrival window plus full "
    "drain, FIFO execution, fresh full nine-option K4096 same-state shadows, "
    "full nine-option K4096 executed references, paired flow endpoint, and "
    "registered compute gates. A PASS authorizes only fixed-protocol action "
    "identity, flow parity/noninferiority, and strict reduction in the three "
    "registered B&B counters (LP calls, LP matrix coefficients, and candidate "
    "evaluations); quotient preprocessing is reported separately. It does not "
    "establish flow improvement, wall-time or total-runtime improvement, or the "
    "continuation branch unless at least one trigger is observed and resolved. "
    "It does not authorize a paper edit or a broader stability, capacity, "
    "physical-hardware, or all-queue claim."
)


def _registration_body(
    paths: Mapping[str, str],
    contract: Mapping[str, Any],
    source_protocol: Mapping[str, Any],
    capacity: Mapping[str, Any],
    eta: Mapping[str, Any],
    quotient: Mapping[str, Any],
) -> dict[str, Any]:
    model = source_protocol.get("physical_complete_frame_model") or {}
    if (
        len(model.get("options") or ()) != ORIGINAL_OPTION_COUNT
        or source_protocol.get("arrival_seeds")
        != list(development.ARRIVAL_SEEDS)
        or source_protocol.get("next_confirmation_seeds") != list(ARRIVAL_SEEDS)
        or source_protocol.get("arrival_horizon_s") != ARRIVAL_HORIZON_S
        or source_protocol.get("method_contract") != METHOD_CONTRACT
        or source_protocol.get("quotient_contract") != quotient
        or set(capacity) != {"8", "16"}
        or set(eta) != {"8", "16"}
    ):
        raise RequirementDominanceConfirmationRefusal(
            "development model, method, quotient, or envelope changed"
        )
    return {
        "schema_version": 1,
        "gate": GATE,
        "experiment": EXPERIMENT,
        "status": REGISTRATION_STATUS,
        "registered": True,
        "independent_confirmation": True,
        "source_paths": dict(paths),
        "development_pass_contract": _json(contract),
        "physical_complete_frame_model": _json(model),
        "physical_envelope_capacity_normalized_loads": _json(capacity),
        "lane_minimum_effective_slack_eta": _json(eta),
        "trajectory_contract": _json(TRAJECTORY_CONTRACT),
        "method_contract": METHOD_CONTRACT,
        "quotient_contract": _json(quotient),
        "work_fields": list(WORK_FIELDS),
        "work_gate_fields": list(WORK_GATE_FIELDS),
        "scientific_gates": {
            "execution_completeness_certificate_and_fifo": (
                "every registered execution-integrity check must pass"
            ),
            "dominance_lift": (
                "every candidate frame must use the prepared eight-option "
                "quotient and have a valid full-catalogue certificate lift"
            ),
            "support": "eta exceeds maximum candidate epsilon and actual regret",
            "action_and_regret": (
                "every candidate action matches its exact fresh full-catalogue "
                "K4096 shadow and actual regret is bounded by the certificate"
            ),
            "flow": "six-family paired mean-flow upper bounds are at most 1.01",
            "case_compute": (
                "candidate <= same-state full-catalogue K4096 shadow in every "
                "gate field"
            ),
            "lane_compute": (
                "candidate < same-state full-catalogue K4096 shadow in every "
                "gate field"
            ),
            "aggregate_compute": (
                "candidate < same-state full-catalogue K4096 shadow in every "
                "gate field"
            ),
        },
        "registration_scope": dict(REGISTRATION_SCOPE),
        "claim_boundary": CLAIM_BOUNDARY,
        "paper_claim_authorized": False,
    }


def _prepare_registered_quotient(
    source_protocol: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    model_snapshot = source_protocol.get("physical_complete_frame_model") or {}
    model = raw.old_transition.baseline._model_from_snapshot(model_snapshot)
    if _json(model.snapshot()) != model_snapshot:
        raise RequirementDominanceConfirmationRefusal(
            "frozen model reconstruction failed"
        )
    prepared = prepare_canonical_count_space_bnb_requirement_dominance_catalogue(
        model.options
    )
    try:
        quotient = development._quotient_contract(prepared)
    except (RuntimeError, ValueError) as exc:
        raise RequirementDominanceConfirmationRefusal(
            f"prepared dominance quotient changed: {exc}"
        ) from exc
    return prepared, quotient


def build_registration(**kwargs: Any) -> dict[str, Any]:
    _validate_seed_roster()
    paths = _source_paths(**kwargs)
    raw_source_protocol = _read_object(Path(paths["development_registration"]))
    _, quotient = _prepare_registered_quotient(raw_source_protocol)
    contract, source_protocol, capacity, eta = _validated_development_bundle(
        paths, quotient
    )
    return _registration_body(
        paths, contract, source_protocol, capacity, eta, quotient
    )


def _validate_protocol(protocol: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    observed = dict(protocol)
    registered_at = observed.pop("registered_at_utc", None)
    if not isinstance(registered_at, str) or observed != expected:
        raise RequirementDominanceConfirmationRefusal("registered protocol changed")


def register(
    registration: Path = DEFAULT_REGISTRATION,
    fixed_sources: Path = DEFAULT_FIXED_SOURCES,
    result: Path = DEFAULT_RESULT,
    **kwargs: Any,
) -> dict[str, Any]:
    if any(Path(path).exists() for path in (registration, fixed_sources, result)):
        raise RequirementDominanceConfirmationRefusal(
            "confirmation artifact already exists"
        )
    body = build_registration(**kwargs)
    protocol = {"registered_at_utc": datetime.now(timezone.utc).isoformat(), **body}
    source_fixed = Path(body["source_paths"]["development_fixed_sources"])
    _write_new(Path(fixed_sources), _current_fixed_sources(source_fixed))
    _write_new(Path(registration), protocol, compact=True)
    return protocol


def _execute_tasks(tasks: Sequence[tuple[Any, ...]]) -> list[dict[str, Any]]:
    with ProcessPoolExecutor(max_workers=EXECUTION_WORKERS) as pool:
        return list(pool.map(development._run_tape, tasks))


def _case_summary(
    runs: Sequence[Mapping[str, Any]], lanes: int, load: float, eta: float
) -> dict[str, Any]:
    summary = confirmation._paired_case_summary(
        runs, lanes, load, eta, ARRIVAL_SEEDS
    )
    members = [row for row in runs if row["lanes"] == lanes and row["load"] == load]
    metrics = [
        next(item for item in row["policies"] if item["policy"] == "candidate")[
            "metrics"
        ]
        for row in members
    ]
    summary.pop("candidate_fixed_budget", None)
    summary.update(
        {
            "candidate_method": (
                "eight_option_requirement_dominance_score_exact_tie_continuation"
            ),
            "candidate_bnb_option_count": QUOTIENT_OPTION_COUNT,
            "shadow_and_reference_option_count": ORIGINAL_OPTION_COUNT,
            "primary_budget": PRIMARY_BUDGETS[lanes],
            "continuation_budget": CONTINUATION_BUDGETS[lanes],
            "tie_continuation_trigger_count": sum(
                row["tie_continuation_trigger_count"] for row in metrics
            ),
            "tie_continuation_final_exact_count": sum(
                row["tie_continuation_final_exact_count"] for row in metrics
            ),
            "conditional_selector_relation_valid_count": sum(
                row["conditional_selector_relation_valid_count"] for row in metrics
            ),
            "requirement_dominance_quotient_selector_call_count": sum(
                row["requirement_dominance_quotient_selector_call_count"]
                for row in metrics
            ),
            "full_catalogue_certificate_lift_valid_count": sum(
                row["full_catalogue_certificate_lift_valid_count"] for row in metrics
            ),
        }
    )
    return summary


def _decision(
    execution_passed: bool,
    cases: Sequence[Mapping[str, Any]],
    compute_passed: bool,
) -> str:
    if not execution_passed:
        return INVALID_DECISION
    if not all(row["trajectory_support_gate_passed"] for row in cases):
        return SUPPORT_FAIL_DECISION
    if not all(row["same_state_action_and_regret_gate_passed"] for row in cases):
        return ACTION_FAIL_DECISION
    if not compute_passed:
        return COMPUTE_FAIL_DECISION
    if not all(row["paired_mean_flow_noninferiority_passed"] for row in cases):
        return FLOW_FAIL_DECISION
    return PASS_DECISION


def run_registered(
    registration: Path = DEFAULT_REGISTRATION,
    fixed_sources: Path = DEFAULT_FIXED_SOURCES,
    output: Path = DEFAULT_RESULT,
) -> dict[str, Any]:
    if Path(output).exists():
        raise RequirementDominanceConfirmationRefusal(f"result exists: {output}")
    protocol = _read_object(Path(registration))
    paths = protocol.get("source_paths") or {}

    # This is the sole quotient preparation in an execution.  The resulting
    # object is passed unchanged to every worker task.
    prepared, quotient = _prepare_registered_quotient(protocol)
    source_contract, source_protocol, capacity, eta = (
        _validated_development_bundle(paths, quotient)
    )
    expected = _registration_body(
        paths, source_contract, source_protocol, capacity, eta, quotient
    )
    _validate_protocol(protocol, expected)
    if source_contract != protocol["development_pass_contract"]:
        raise RequirementDominanceConfirmationRefusal(
            "development PASS contract changed"
        )
    registered_sources = _read_object(Path(fixed_sources))
    current_sources = _current_fixed_sources(
        Path(paths["development_fixed_sources"])
    )
    if registered_sources != current_sources:
        raise RequirementDominanceConfirmationRefusal("registered sources changed")

    model = raw.old_transition.baseline._model_from_snapshot(
        protocol["physical_complete_frame_model"]
    )
    if _json(model.snapshot()) != protocol["physical_complete_frame_model"]:
        raise RequirementDominanceConfirmationRefusal(
            "frozen execution model reconstruction failed"
        )
    capacity = protocol["physical_envelope_capacity_normalized_loads"]
    by_case: dict[tuple[int, float], Mapping[str, Any]] = {}
    for lanes in LANES:
        for row in capacity[str(lanes)]:
            key = (lanes, float(row["load"]))
            if (
                int(row["lanes"]) != lanes
                or row.get("capacity_projection_passed") is not True
                or row.get("conditional_negative_drift_certified") is not True
                or row.get("positive_effective_slack") is not True
            ):
                raise RequirementDominanceConfirmationRefusal(
                    "capacity row changed"
                )
            by_case[key] = row
    if set(by_case) != {(lanes, load) for lanes in LANES for load in LOADS}:
        raise RequirementDominanceConfirmationRefusal("capacity grid changed")
    tasks = [
        (
            model,
            by_case[(lanes, load)],
            prepared,
            lanes,
            load,
            seed,
            ARRIVAL_HORIZON_S,
        )
        for lanes in LANES
        for load in LOADS
        for seed in ARRIVAL_SEEDS
    ]
    started = time.perf_counter()
    runs = _execute_tasks(tasks)

    post_contract, post_protocol, post_capacity, post_eta = (
        _validated_development_bundle(paths, quotient)
    )
    post_expected = _registration_body(
        paths,
        post_contract,
        post_protocol,
        post_capacity,
        post_eta,
        quotient,
    )
    _validate_protocol(protocol, post_expected)
    if (
        _read_object(Path(fixed_sources)) != registered_sources
        or registered_sources
        != _current_fixed_sources(Path(paths["development_fixed_sources"]))
    ):
        raise RequirementDominanceConfirmationRefusal(
            "registered sources changed during execution"
        )

    eta = protocol["lane_minimum_effective_slack_eta"]
    cases = [
        _case_summary(runs, lanes, load, float(eta[str(lanes)]))
        for lanes in LANES
        for load in LOADS
    ]
    work_gates = raw._aggregate_work_gates(runs, cases)
    policy_rows = [policy for run in runs for policy in run["policies"]]
    candidate_rows = [row for row in policy_rows if row["policy"] == "candidate"]
    reference_rows = [
        row for row in policy_rows if row["policy"] == "cold_k4096_reference"
    ]
    candidate_metrics = [row["metrics"] for row in candidate_rows]
    total_candidate_frames = sum(row["frame_count"] for row in candidate_metrics)
    total_reference_frames = sum(
        row["metrics"]["frame_count"] for row in reference_rows
    )
    trigger_count = sum(
        row["tie_continuation_trigger_count"] for row in candidate_metrics
    )
    final_exact_count = sum(
        row["tie_continuation_final_exact_count"] for row in candidate_metrics
    )
    expected_run_keys = [
        (lanes, load, seed)
        for lanes in LANES
        for load in LOADS
        for seed in ARRIVAL_SEEDS
    ]
    observed_run_keys = [
        (int(row["lanes"]), float(row["load"]), int(row["seed"]))
        for row in runs
    ]
    checks = {
        "development_pass_and_65_sources_revalidated": True,
        "development_current_normal_sources_have_zero_drift": (
            source_contract["current_normal_source_drift"] == []
        ),
        "protocol_and_66_sources_revalidated_after_execution": True,
        "independent_seed_roster_is_disjoint_from_all_173_prior_seeds": not (
            set(ARRIVAL_SEEDS).intersection(PREVIOUSLY_USED_ARRIVAL_SEEDS)
        ),
        "all_120_registered_case_seed_tapes_execute_exactly_once": (
            observed_run_keys == expected_run_keys and len(runs) == 120
        ),
        "executed_trajectory_count_is_240": len(policy_rows) == 240,
        "each_tape_executes_candidate_and_full_reference_once": all(
            [row["policy"] for row in run["policies"]]
            == ["candidate", "cold_k4096_reference"]
            and run["arrival_tape"]["executed_policy_count"] == 2
            for run in runs
        ),
        "one_arrival_generation_per_common_tape": all(
            run["arrival_tape"]["arrival_generation_count"] == 1 for run in runs
        ),
        "same_state_shadow_never_executes_a_trajectory": all(
            run["arrival_tape"]["shadow_trajectory_runs"] == 0 for run in runs
        ),
        "one_prepared_quotient_call_per_candidate_frame": all(
            row["requirement_dominance_quotient_selector_call_count"]
            == row["frame_count"]
            for row in candidate_metrics
        ),
        "every_full_catalogue_certificate_lift_is_valid": all(
            row["full_catalogue_certificate_lift_valid_count"]
            == row["frame_count"]
            for row in candidate_metrics
        ),
        "every_conditional_selector_relation_is_valid": all(
            row["conditional_selector_relation_valid_count"] == row["frame_count"]
            for row in candidate_metrics
        ),
        "trigger_is_exactly_primary_score_exact_noncanonical": all(
            row["tie_continuation_trigger_count"]
            == row["primary_score_exact_noncanonical_count"]
            for row in candidate_metrics
        ),
        "every_candidate_frame_has_one_fresh_full_k4096_shadow": all(
            row["same_state_shadow_calls"]
            == row["same_state_shadow_valid_count"]
            == row["same_state_shadow_exact_count"]
            == row["frame_count"]
            for row in candidate_metrics
        ),
        "candidate_and_reference_catalogues_remain_eight_and_nine_options": all(
            row["candidate_bnb_option_count"] == QUOTIENT_OPTION_COUNT
            and row["shadow_and_reference_option_count"] == ORIGINAL_OPTION_COUNT
            for row in cases
        ),
        "every_policy_fully_drains_and_flow_decomposes": all(
            row["flow_decomposition"]["status"] == "PASS" for row in policy_rows
        ),
        "every_case_contains_the_registered_20_seed_roster": all(
            row["seeds"] == list(ARRIVAL_SEEDS) and row["seed_count"] == 20
            for row in cases
        ),
        "every_case_execution_integrity_and_certificate_gate_passes": all(
            row["execution_integrity_passed"] for row in cases
        ),
        "no_capacity_recomputation_reclassification_or_server_pull": True,
    }
    execution_passed = all(checks.values())
    decision = _decision(execution_passed, cases, work_gates["gate_passed"])
    status = (
        "PASS"
        if decision == PASS_DECISION
        else ("INVALID" if decision == INVALID_DECISION else "FAIL")
    )
    passed = decision == PASS_DECISION
    continuation_branch_observed = trigger_count > 0
    continuation_branch_empirically_confirmed = bool(
        continuation_branch_observed and trigger_count == final_exact_count
    )
    report = {
        "schema_version": 1,
        "gate": GATE,
        "experiment": EXPERIMENT,
        "status": status,
        "scientific_decision": decision,
        "protocol": protocol,
        "development_pass_contract": source_contract,
        "checks": checks,
        "case_summaries": cases,
        "runs": runs,
        "work_gates": work_gates,
        "tie_continuation_trigger_count": trigger_count,
        "tie_continuation_final_exact_count": final_exact_count,
        "continuation_branch_observed": continuation_branch_observed,
        "continuation_branch_empirically_confirmed": (
            continuation_branch_empirically_confirmed
        ),
        "catalogue_preprocessing_work": protocol["quotient_contract"]["reduction"],
        "work_accounting": {
            "arrival_tapes": len(runs),
            "arrival_generations": len(runs),
            "executed_trajectory_runs": len(policy_rows),
            "candidate_trajectory_runs": len(candidate_rows),
            "cold_full_catalogue_reference_trajectory_runs": len(reference_rows),
            "shadow_trajectory_runs": 0,
            "quotient_candidate_selector_calls": total_candidate_frames,
            "same_state_full_catalogue_shadow_selector_calls": total_candidate_frames,
            "executed_full_catalogue_reference_selector_calls": (
                total_reference_frames
            ),
            "cold_restart_rescue_calls": 0,
            "server_pulls": 0,
        },
        "independent_confirmation_passed": passed,
        "trajectory_transition_passed": passed,
        "trajectory_or_flow_claim_authorized": passed,
        "fixed_protocol_action_identity_and_flow_noninferiority_authorized": passed,
        "registered_bnb_counter_reduction_authorized": passed,
        "continuation_branch_claim_authorized": bool(
            passed and continuation_branch_empirically_confirmed
        ),
        "flow_improvement_claim_authorized": False,
        "wall_time_or_total_runtime_claim_authorized": False,
        "fixed_protocol_claim_only": True,
        "paper_claim_authorized": False,
        "trajectory_runs": len(policy_rows),
        "arrival_generations": len(runs),
        "shadow_trajectory_runs": 0,
        "server_pulls": 0,
        "claim_boundary": CLAIM_BOUNDARY,
        "timing": {"wall_s": time.perf_counter() - started},
    }
    _write_new(Path(output), report, compact=True)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--register", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    parser.add_argument("--fixed-sources", type=Path, default=DEFAULT_FIXED_SOURCES)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    args = parser.parse_args(argv)
    try:
        payload = (
            register(args.registration, args.fixed_sources, args.output)
            if args.register
            else run_registered(args.registration, args.fixed_sources, args.output)
        )
    except RequirementDominanceConfirmationRefusal as exc:
        raise SystemExit(f"REFUSED: {exc}") from exc
    print(
        json.dumps(
            {
                "status": payload["status"],
                "experiment": payload["experiment"],
                "scientific_decision": payload.get("scientific_decision"),
                "tie_continuation_trigger_count": payload.get(
                    "tie_continuation_trigger_count"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
