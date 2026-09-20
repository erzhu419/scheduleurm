"""Post-result trajectory development for certificate-triggered tie continuation.

The predecessor raw fixed-cap trajectory is retained as a formal FAIL.  This
development replay reuses its 120 already observed tapes and changes only the
online selector: after the registered R8/K288 or R16/K576 primary checkpoint,
one live B&B frontier continues to R8/K576 or R16/K1152 exactly when the score
is proved exact but the canonical tie-break is not.  A passing replay only
authorizes a later confirmation on disjoint arrival seeds.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from algorithm.canonical_count_space_bnb_frame_candidates import (
    select_canonical_count_space_bnb_score_exact_tie_continuation_frame,
)
from algorithm.experiments import (
    aligned_hardware_joint95_v5_envelope_raw_fixed_cap_trajectory_transition
    as raw,
)
from algorithm.experiments.aligned_frame_flow_decomposition import decompose_flow
from algorithm.experiments.aligned_hardware_frame_experiment import poisson_arrivals


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
STEM = (
    "aligned_hardware_joint95_v5_envelope_"
    "tie_continuation_development_20260920"
)
DEFAULT_REGISTRATION = ARTIFACTS / f"{STEM}_protocol.json"
DEFAULT_FIXED_SOURCES = ARTIFACTS / f"{STEM}_fixed_sources.json"
DEFAULT_RESULT = ARTIFACTS / f"{STEM}.json"
DEFAULT_PREDECESSOR_REGISTRATION = raw.DEFAULT_REGISTRATION
DEFAULT_PREDECESSOR_FIXED_SOURCES = raw.DEFAULT_FIXED_SOURCES
DEFAULT_PREDECESSOR_RESULT = raw.DEFAULT_RESULT

EXPERIMENT = (
    "aligned_hardware_joint95_v5_envelope_"
    "tie_continuation_development_v1"
)
GATE = (
    "aligned_hardware_joint95_v5_envelope_"
    "tie_continuation_development_runner"
)
REGISTRATION_STATUS = "REGISTERED_AWAITING_TIE_CONTINUATION_DEVELOPMENT_REPLAY"
PASS_DECISION = "TIE_CONTINUATION_TRAJECTORY_DEVELOPMENT_PASS"
INVALID_DECISION = "INVALID_TIE_CONTINUATION_DEVELOPMENT_EXECUTION"
SUPPORT_FAIL_DECISION = "TIE_CONTINUATION_DEVELOPMENT_SUPPORT_FAIL_STOP"
ACTION_FAIL_DECISION = "TIE_CONTINUATION_DEVELOPMENT_ACTION_OR_REGRET_FAIL_STOP"
COMPUTE_FAIL_DECISION = "TIE_CONTINUATION_DEVELOPMENT_COMPUTE_FAIL_STOP"
FLOW_FAIL_DECISION = "TIE_CONTINUATION_DEVELOPMENT_FLOW_NONINFERIORITY_FAIL_STOP"

LANES = raw.LANES
LOADS = raw.LOADS
ARRIVAL_SEEDS = raw.ARRIVAL_SEEDS
ARRIVAL_HORIZON_S = raw.ARRIVAL_HORIZON_S
PRIMARY_BUDGETS = {8: 288, 16: 576}
CONTINUATION_BUDGETS = {8: 576, 16: 1152}
REFERENCE_BUDGET = raw.REFERENCE_BUDGET
WORK_FIELDS = raw.WORK_FIELDS
WORK_GATE_FIELDS = raw.WORK_GATE_FIELDS
SCORE_TOLERANCE = raw.SCORE_TOLERANCE
EXECUTION_WORKERS = raw.EXECUTION_WORKERS
THIS_SOURCE = (
    "algorithm/experiments/"
    "aligned_hardware_joint95_v5_envelope_"
    "tie_continuation_development.py"
)
CORE_SOURCE = "algorithm/canonical_count_space_bnb_frame_candidates.py"
HISTORICAL_SOURCE_PREFIX = "__historical_predecessor__/"


class TieContinuationDevelopmentRefusal(RuntimeError):
    pass


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TieContinuationDevelopmentRefusal(f"expected JSON object: {path}")
    return payload


def _write_new(path: Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=None if compact else 2, sort_keys=True)
        handle.write("\n")


def _source_paths(
    predecessor_registration: Path = DEFAULT_PREDECESSOR_REGISTRATION,
    predecessor_fixed_sources: Path = DEFAULT_PREDECESSOR_FIXED_SOURCES,
    predecessor_result: Path = DEFAULT_PREDECESSOR_RESULT,
) -> dict[str, str]:
    return {
        "predecessor_registration": str(Path(predecessor_registration).resolve()),
        "predecessor_fixed_sources": str(Path(predecessor_fixed_sources).resolve()),
        "predecessor_result": str(Path(predecessor_result).resolve()),
    }


def _current_fixed_sources(
    predecessor_fixed_sources: Path = DEFAULT_PREDECESSOR_FIXED_SOURCES,
) -> dict[str, str]:
    historical = _read_object(Path(predecessor_fixed_sources))
    current: dict[str, str] = {}
    for relative, frozen_text in historical.items():
        current[f"{HISTORICAL_SOURCE_PREFIX}{relative}"] = frozen_text
        source = ROOT / relative
        current[relative] = (
            source.read_text(encoding="utf-8") if source.is_file() else frozen_text
        )
    current[THIS_SOURCE] = (ROOT / THIS_SOURCE).read_text(encoding="utf-8")
    return dict(sorted(current.items()))


def _validated_predecessor_bundle(
    paths: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if set(paths) != {
        "predecessor_registration",
        "predecessor_fixed_sources",
        "predecessor_result",
    }:
        raise TieContinuationDevelopmentRefusal("predecessor paths changed")
    protocol = _read_object(Path(paths["predecessor_registration"]))
    frozen = _read_object(Path(paths["predecessor_fixed_sources"]))
    result = _read_object(Path(paths["predecessor_result"]))
    if (
        result.get("protocol") != protocol
        or result.get("gate") != raw.GATE
        or result.get("experiment") != raw.EXPERIMENT
        or result.get("status") != "FAIL"
        or result.get("scientific_decision") != raw.ACTION_FAIL_DECISION
        or result.get("trajectory_transition_passed") is not False
        or result.get("paper_claim_authorized") is not False
        or result.get("trajectory_runs") != 240
        or result.get("arrival_generations") != 120
        or result.get("server_pulls") != 0
        or not result.get("checks")
        or not all(result["checks"].values())
        or not (result.get("work_gates") or {}).get("gate_passed")
    ):
        raise TieContinuationDevelopmentRefusal(
            "predecessor is not the required valid action-identity FAIL"
        )
    if (
        len(frozen) != 30
        or raw.THIS_SOURCE not in frozen
        or CORE_SOURCE not in frozen
        or frozen[raw.THIS_SOURCE]
        != (ROOT / raw.THIS_SOURCE).read_text(encoding="utf-8")
    ):
        raise TieContinuationDevelopmentRefusal(
            "predecessor historical fixed-source bundle changed"
        )
    drift = sorted(
        relative
        for relative, frozen_text in frozen.items()
        if (ROOT / relative).is_file()
        and (ROOT / relative).read_text(encoding="utf-8") != frozen_text
    )
    historical_core = frozen[CORE_SOURCE]
    current_core = (ROOT / CORE_SOURCE).read_text(encoding="utf-8")
    if (
        drift != [CORE_SOURCE]
        or "def trace_canonical_count_space_bnb_frame(" not in historical_core
        or "select_canonical_count_space_bnb_score_exact_tie_continuation_frame"
        in historical_core
        or "select_canonical_count_space_bnb_score_exact_tie_continuation_frame"
        not in current_core
    ):
        raise TieContinuationDevelopmentRefusal(
            "expected only the registered core-selector successor change"
        )

    total_frames = action_mismatches = regret_violations = 0
    mismatch_rows: list[dict[str, Any]] = []
    for run in result.get("runs") or ():
        policies = {row["policy"]: row for row in run["policies"]}
        metrics = policies["candidate"]["metrics"]
        frames = int(metrics["frame_count"])
        mismatches = frames - int(metrics["same_state_action_match_count"])
        violations = frames - int(metrics["regret_bounded_by_epsilon_count"])
        total_frames += frames
        action_mismatches += mismatches
        regret_violations += violations
        if mismatches:
            mismatch_rows.append(
                {
                    "lanes": run["lanes"],
                    "load": run["load"],
                    "seed": run["seed"],
                    **metrics["first_action_mismatch"],
                    "mismatch_count": mismatches,
                }
            )
    expected_mismatch = {
        "lanes": 16,
        "load": 0.8,
        "seed": 12077,
        "frame": 12,
        "queue": [25, 3, 23],
        "candidate": [0, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 5, 7, 7, 8, 8],
        "shadow": [0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 5, 8, 8],
        "regret": 0.0,
        "mismatch_count": 1,
    }
    cases = result.get("case_summaries") or ()
    if (
        total_frames != 5365
        or action_mismatches != 1
        or regret_violations != 0
        or mismatch_rows != [expected_mismatch]
        or len(cases) != 6
        or sum(not row["same_state_action_and_regret_gate_passed"] for row in cases)
        != 1
        or not all(
            row["execution_integrity_passed"]
            and row["trajectory_support_gate_passed"]
            and row["paired_mean_flow_noninferiority_passed"]
            and row["case_compute_gate_passed"]
            for row in cases
        )
    ):
        raise TieContinuationDevelopmentRefusal(
            "predecessor failure structure changed"
        )
    contract = {
        "experiment": result["experiment"],
        "gate": result["gate"],
        "status": result["status"],
        "scientific_decision": result["scientific_decision"],
        "trajectory_runs": result["trajectory_runs"],
        "arrival_generations": result["arrival_generations"],
        "candidate_frame_count": total_frames,
        "action_mismatch_count": action_mismatches,
        "regret_violation_count": regret_violations,
        "only_action_mismatch": expected_mismatch,
        "all_execution_checks_passed": True,
        "all_non_action_scientific_gates_passed": True,
        "historical_fixed_source_keys": sorted(frozen),
        "current_source_drift_from_predecessor": drift,
        "historical_core_source_is_retained_in_predecessor_bundle": True,
        "successor_fixed_sources_retain_historical_bundle_under_prefix": (
            HISTORICAL_SOURCE_PREFIX
        ),
    }
    capacity = protocol.get("physical_envelope_capacity_normalized_loads")
    eta = protocol.get("lane_minimum_effective_slack_eta")
    if not isinstance(capacity, dict) or not isinstance(eta, dict):
        raise TieContinuationDevelopmentRefusal("predecessor model envelope missing")
    return contract, protocol, _json(capacity), _json(eta)


METHOD_CONTRACT = {
    "primary_budget_by_lane": {str(k): v for k, v in PRIMARY_BUDGETS.items()},
    "continuation_budget_by_lane": {
        str(k): v for k, v in CONTINUATION_BUDGETS.items()
    },
    "trigger": "primary_score_exact_and_not_primary_canonical_exact",
    "continuation": "resume_the_same_live_bnb_frontier",
    "cold_restart_calls": 0,
    "shadow_information_read_by_candidate": False,
    "load_seed_or_future_information_read_by_candidate": False,
    "unresolved_at_continuation_cap": "return_the_capped_selection_and_apply_frozen_gates",
}


REGISTRATION_SCOPE = {
    "post_result_development": True,
    "reuses_predecessor_tapes_12061_through_12080": True,
    "independent_confirmation": False,
    "rng_initializations": 0,
    "rng_draws": 0,
    "arrival_generations": 0,
    "selector_calls": 0,
    "trajectory_runs": 0,
    "server_pulls": 0,
    "paper_or_readme_edits": 0,
}


CLAIM_BOUNDARY = (
    "This is a post-result development replay on the already observed 12061--12080 "
    "arrival tapes. A PASS only authorizes implementation and a later trajectory "
    "confirmation on disjoint arrival seeds. It is not independent evidence and "
    "does not authorize a trajectory, flow-time, stability, capacity, physical "
    "performance, or paper claim."
)


def build_registration(**kwargs: Any) -> dict[str, Any]:
    paths = _source_paths(**kwargs)
    contract, source_protocol, capacity, eta = _validated_predecessor_bundle(paths)
    model = source_protocol.get("physical_complete_frame_model") or {}
    trajectory_contract = source_protocol.get("trajectory_contract") or {}
    if (
        len(model.get("options") or ()) != 9
        or trajectory_contract.get("arrival_seeds") != list(ARRIVAL_SEEDS)
        or trajectory_contract.get("case_seed_tapes") != 120
        or trajectory_contract.get("trajectory_runs") != 240
        or set(capacity) != {"8", "16"}
        or set(eta) != {"8", "16"}
    ):
        raise TieContinuationDevelopmentRefusal(
            "predecessor model, seed roster or envelope changed"
        )
    return {
        "schema_version": 1,
        "gate": GATE,
        "experiment": EXPERIMENT,
        "status": REGISTRATION_STATUS,
        "registered": True,
        "source_paths": paths,
        "predecessor_fail_contract": contract,
        "physical_complete_frame_model": _json(model),
        "physical_envelope_capacity_normalized_loads": capacity,
        "lane_minimum_effective_slack_eta": eta,
        "arrival_seeds": list(ARRIVAL_SEEDS),
        "arrival_horizon_s": ARRIVAL_HORIZON_S,
        "method_contract": _json(METHOD_CONTRACT),
        "work_fields": list(WORK_FIELDS),
        "work_gate_fields": list(WORK_GATE_FIELDS),
        "scientific_gates": {
            "execution_action_support_regret_certificate_and_flow": (
                "identical to the predecessor trajectory protocol"
            ),
            "case_compute": "candidate <= same-state K4096 shadow in every gate field",
            "lane_compute": "candidate < same-state K4096 shadow in every gate field",
            "aggregate_compute": "candidate < same-state K4096 shadow in every gate field",
        },
        "registration_scope": dict(REGISTRATION_SCOPE),
        "claim_boundary": CLAIM_BOUNDARY,
        "paper_claim_authorized": False,
    }


def _validate_protocol(protocol: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    observed = dict(protocol)
    registered_at = observed.pop("registered_at_utc", None)
    if not isinstance(registered_at, str) or observed != expected:
        raise TieContinuationDevelopmentRefusal("registered protocol changed")


def register(
    registration: Path = DEFAULT_REGISTRATION,
    fixed_sources: Path = DEFAULT_FIXED_SOURCES,
    result: Path = DEFAULT_RESULT,
    **kwargs: Any,
) -> dict[str, Any]:
    if any(Path(path).exists() for path in (registration, fixed_sources, result)):
        raise TieContinuationDevelopmentRefusal("development artifact already exists")
    body = build_registration(**kwargs)
    protocol = {"registered_at_utc": datetime.now(timezone.utc).isoformat(), **body}
    source_fixed = Path(body["source_paths"]["predecessor_fixed_sources"])
    _write_new(Path(fixed_sources), _current_fixed_sources(source_fixed))
    _write_new(Path(registration), protocol, compact=True)
    return protocol


def _candidate_call(model: Any, queue: Sequence[int]) -> dict[str, Any]:
    lanes = int(model.lanes)
    initial_budget = PRIMARY_BUDGETS[lanes]
    continuation_budget = CONTINUATION_BUDGETS[lanes]
    continued = select_canonical_count_space_bnb_score_exact_tie_continuation_frame(
        model.options,
        queue,
        lanes,
        initial_budget,
        continuation_budget,
    )
    primary, selected = continued.primary_selection, continued.selection
    expected_trigger = bool(primary.score_exact and not primary.canonical_exact)
    relation = bool(
        continued.initial_budget == initial_budget
        and continued.continuation_budget == continuation_budget
        and continued.continuation_triggered is expected_trigger
        and (
            (not expected_trigger and selected is primary)
            or (
                expected_trigger
                and selected.lp_node_budget == continuation_budget
                and selected.score + SCORE_TOLERANCE >= primary.score
                and selected.upper_bound
                <= primary.upper_bound + SCORE_TOLERANCE
                and all(
                    getattr(selected, field) >= getattr(primary, field)
                    for field in WORK_FIELDS
                )
            )
        )
    )
    valid = bool(
        relation
        and raw._certificate_valid(
            model, queue, initial_budget, primary, exact=False
        )
        and raw._certificate_valid(
            model,
            queue,
            selected.lp_node_budget,
            selected,
            exact=False,
        )
    )
    return {
        "selection": selected,
        "primary_selection": primary,
        "work": raw._selection_work(selected),
        "valid": valid,
        "method_relation_valid": relation,
        "continuation_triggered": expected_trigger,
        "initial_budget": initial_budget,
        "continuation_budget": continuation_budget,
    }


def _run_tape(
    task: tuple[Any, Mapping[str, Any], int, float, int, float]
) -> dict[str, Any]:
    model, capacity, lanes, load, seed, horizon = task
    lane_model = replace(model, lanes=lanes)
    tape = tuple(poisson_arrivals(capacity["arrival_rates"], horizon, seed))
    tracking = {
        "calls": 0,
        "relations": 0,
        "triggers": 0,
        "trigger_final_exact": 0,
        "primary_score_exact_noncanonical": 0,
        "trigger_audit_prefix": [],
    }

    def tracked_candidate(current_model: Any, queue: Sequence[int]) -> dict[str, Any]:
        call = _candidate_call(current_model, queue)
        primary, selected = call["primary_selection"], call["selection"]
        tracking["calls"] += 1
        tracking["relations"] += int(call["method_relation_valid"])
        tracking["triggers"] += int(call["continuation_triggered"])
        tracking["primary_score_exact_noncanonical"] += int(
            primary.score_exact and not primary.canonical_exact
        )
        tracking["trigger_final_exact"] += int(
            call["continuation_triggered"]
            and selected.score_exact
            and selected.canonical_exact
        )
        if call["continuation_triggered"] and len(tracking["trigger_audit_prefix"]) < 3:
            tracking["trigger_audit_prefix"].append(
                {
                    "queue": list(queue),
                    "primary_action": list(primary.option_indices),
                    "selected_action": list(selected.option_indices),
                    "primary_lp_calls": primary.lp_calls,
                    "selected_lp_calls": selected.lp_calls,
                    "selected_canonical_exact": selected.canonical_exact,
                }
            )
        return call

    previous_candidate = raw._candidate_call
    raw._candidate_call = tracked_candidate
    try:
        candidate_metrics, candidate_frames = raw.simulate_trajectory(
            lane_model, tape, horizon, "candidate"
        )
    finally:
        raw._candidate_call = previous_candidate
    candidate_metrics.update(
        {
            "conditional_selector_call_count": tracking["calls"],
            "conditional_selector_relation_valid_count": tracking["relations"],
            "tie_continuation_trigger_count": tracking["triggers"],
            "tie_continuation_final_exact_count": tracking["trigger_final_exact"],
            "primary_score_exact_noncanonical_count": tracking[
                "primary_score_exact_noncanonical"
            ],
            "tie_continuation_audit_prefix": tracking["trigger_audit_prefix"],
        }
    )
    reference_metrics, reference_frames = raw.simulate_trajectory(
        lane_model, tape, horizon, "cold_k4096_reference"
    )
    candidate_decomposition = decompose_flow(
        lane_model, tape, candidate_frames, horizon
    )
    reference_decomposition = decompose_flow(
        lane_model, tape, reference_frames, horizon
    )
    jobs_by_class = [0] * len(model.workloads)
    for _, coordinate, _ in tape:
        jobs_by_class[coordinate] += 1
    return {
        "lanes": lanes,
        "load": load,
        "seed": seed,
        "arrival_tape": {
            "job_count": len(tape),
            "jobs_by_class": jobs_by_class,
            "first_arrival_s": tape[0][0] if tape else None,
            "last_arrival_s": tape[-1][0] if tape else None,
            "arrival_generation_count": 1,
            "executed_policy_count": 2,
            "shadow_trajectory_runs": 0,
        },
        "policies": [
            {
                "policy": "candidate",
                "metrics": candidate_metrics,
                "flow_decomposition": raw._compact_decomposition(
                    candidate_decomposition
                ),
            },
            {
                "policy": "cold_k4096_reference",
                "metrics": reference_metrics,
                "flow_decomposition": raw._compact_decomposition(
                    reference_decomposition
                ),
            },
        ],
    }


def _case_summary(
    runs: Sequence[Mapping[str, Any]], lanes: int, load: float, eta: float
) -> dict[str, Any]:
    summary = raw._paired_case_summary(runs, lanes, load, eta)
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
            "candidate_method": "score_exact_tie_conditional_continuation",
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
        raise TieContinuationDevelopmentRefusal(f"result exists: {output}")
    protocol = _read_object(Path(registration))
    paths = protocol.get("source_paths") or {}
    source_contract, _, _, _ = _validated_predecessor_bundle(paths)
    expected = build_registration(
        predecessor_registration=Path(paths["predecessor_registration"]),
        predecessor_fixed_sources=Path(paths["predecessor_fixed_sources"]),
        predecessor_result=Path(paths["predecessor_result"]),
    )
    _validate_protocol(protocol, expected)
    if source_contract != protocol["predecessor_fail_contract"]:
        raise TieContinuationDevelopmentRefusal("predecessor FAIL contract changed")
    registered_sources = _read_object(Path(fixed_sources))
    current_sources = _current_fixed_sources(
        Path(paths["predecessor_fixed_sources"])
    )
    if registered_sources != current_sources:
        raise TieContinuationDevelopmentRefusal("registered sources changed")

    model = raw.old_transition.baseline._model_from_snapshot(
        protocol["physical_complete_frame_model"]
    )
    if _json(model.snapshot()) != protocol["physical_complete_frame_model"]:
        raise TieContinuationDevelopmentRefusal("frozen model reconstruction failed")
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
                raise TieContinuationDevelopmentRefusal("capacity row changed")
            by_case[key] = row
    expected_cases = {(lanes, load) for lanes in LANES for load in LOADS}
    if set(by_case) != expected_cases:
        raise TieContinuationDevelopmentRefusal("capacity grid changed")
    tasks = [
        (model, by_case[(lanes, load)], lanes, load, seed, ARRIVAL_HORIZON_S)
        for lanes in LANES
        for load in LOADS
        for seed in ARRIVAL_SEEDS
    ]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=EXECUTION_WORKERS) as pool:
        runs = list(pool.map(_run_tape, tasks))

    post_expected = build_registration(
        predecessor_registration=Path(paths["predecessor_registration"]),
        predecessor_fixed_sources=Path(paths["predecessor_fixed_sources"]),
        predecessor_result=Path(paths["predecessor_result"]),
    )
    _validate_protocol(protocol, post_expected)
    if (
        _read_object(Path(fixed_sources)) != registered_sources
        or registered_sources
        != _current_fixed_sources(Path(paths["predecessor_fixed_sources"]))
    ):
        raise TieContinuationDevelopmentRefusal(
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
    checks = {
        "predecessor_formal_fail_and_historical_sources_revalidated": True,
        "protocol_and_current_sources_revalidated_after_execution": True,
        "development_reuses_registered_120_observed_tapes": len(runs)
        == len(tasks)
        == 120,
        "executed_trajectory_count_is_240": len(policy_rows) == 240,
        "one_arrival_generation_per_common_tape": sum(
            run["arrival_tape"]["arrival_generation_count"] for run in runs
        )
        == 120,
        "same_state_shadow_never_executes_a_trajectory": all(
            run["arrival_tape"]["shadow_trajectory_runs"] == 0 for run in runs
        ),
        "one_conditional_selector_call_per_candidate_frame": sum(
            row["conditional_selector_call_count"] for row in candidate_metrics
        )
        == total_candidate_frames,
        "every_conditional_selector_relation_is_valid": sum(
            row["conditional_selector_relation_valid_count"]
            for row in candidate_metrics
        )
        == total_candidate_frames,
        "trigger_is_exactly_primary_score_exact_noncanonical": trigger_count
        == sum(
            row["primary_score_exact_noncanonical_count"]
            for row in candidate_metrics
        ),
        "every_candidate_frame_has_one_fresh_k4096_shadow": sum(
            row["same_state_shadow_calls"] for row in candidate_metrics
        )
        == total_candidate_frames,
        "every_policy_fully_drains_and_flow_decomposes": all(
            row["flow_decomposition"]["status"] == "PASS" for row in policy_rows
        ),
        "every_case_execution_integrity_passes": all(
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
    report = {
        "schema_version": 1,
        "gate": GATE,
        "experiment": EXPERIMENT,
        "status": status,
        "scientific_decision": decision,
        "protocol": protocol,
        "predecessor_fail_contract": source_contract,
        "checks": checks,
        "case_summaries": cases,
        "runs": runs,
        "work_gates": work_gates,
        "tie_continuation_trigger_count": trigger_count,
        "tie_continuation_final_exact_count": sum(
            row["tie_continuation_final_exact_count"] for row in candidate_metrics
        ),
        "all_triggered_development_states_reached_canonical_exact": trigger_count
        == sum(
            row["tie_continuation_final_exact_count"] for row in candidate_metrics
        ),
        "work_accounting": {
            "arrival_tapes": len(runs),
            "arrival_generations": len(runs),
            "executed_trajectory_runs": len(policy_rows),
            "candidate_trajectory_runs": len(candidate_rows),
            "cold_reference_trajectory_runs": len(reference_rows),
            "shadow_trajectory_runs": 0,
            "conditional_candidate_selector_calls": total_candidate_frames,
            "same_state_shadow_selector_calls": total_candidate_frames,
            "executed_reference_selector_calls": total_reference_frames,
            "cold_restart_rescue_calls": 0,
            "server_pulls": 0,
        },
        "development_replay_passed": decision == PASS_DECISION,
        "new_seed_independent_confirmation_authorized": decision == PASS_DECISION,
        "independent_confirmation_eligible": False,
        "trajectory_or_flow_claim_authorized": False,
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
    except TieContinuationDevelopmentRefusal as exc:
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
