"""Independent new-seed trajectory confirmation of tie continuation.

Registration reads the successful post-result development replay and freezes
fresh arrival seeds 12081--12100 before any arrival generation, selector call,
or trajectory execution.  Execution reuses the frozen development method and
its FIFO trajectory implementation.  Every candidate state is audited by one
separate cold K4096 canonical shadow.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import mean, stdev
import time
from typing import Any, Mapping, Sequence

from scipy.stats import t as student_t

from algorithm.experiments import (
    aligned_hardware_joint95_v5_envelope_tie_continuation_development
    as development,
)


raw = development.raw
ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
STEM = (
    "aligned_hardware_joint95_v5_envelope_"
    "tie_continuation_confirmation_20260920"
)
DEFAULT_REGISTRATION = ARTIFACTS / f"{STEM}_protocol.json"
DEFAULT_FIXED_SOURCES = ARTIFACTS / f"{STEM}_fixed_sources.json"
DEFAULT_RESULT = ARTIFACTS / f"{STEM}.json"
DEFAULT_DEVELOPMENT_REGISTRATION = development.DEFAULT_REGISTRATION
DEFAULT_DEVELOPMENT_FIXED_SOURCES = development.DEFAULT_FIXED_SOURCES
DEFAULT_DEVELOPMENT_RESULT = development.DEFAULT_RESULT

EXPERIMENT = (
    "aligned_hardware_joint95_v5_envelope_"
    "tie_continuation_confirmation_v1"
)
GATE = (
    "aligned_hardware_joint95_v5_envelope_"
    "tie_continuation_confirmation_runner"
)
REGISTRATION_STATUS = "REGISTERED_AWAITING_INDEPENDENT_TIE_CONTINUATION_TRAJECTORY"
PASS_DECISION = "INDEPENDENT_TIE_CONTINUATION_TRAJECTORY_PASS"
INVALID_DECISION = "INVALID_TIE_CONTINUATION_CONFIRMATION_EXECUTION"
SUPPORT_FAIL_DECISION = "TIE_CONTINUATION_CONFIRMATION_SUPPORT_FAIL_STOP"
ACTION_FAIL_DECISION = "TIE_CONTINUATION_CONFIRMATION_ACTION_OR_REGRET_FAIL_STOP"
COMPUTE_FAIL_DECISION = "TIE_CONTINUATION_CONFIRMATION_COMPUTE_FAIL_STOP"
FLOW_FAIL_DECISION = "TIE_CONTINUATION_CONFIRMATION_FLOW_NONINFERIORITY_FAIL_STOP"

LANES = development.LANES
LOADS = development.LOADS
ARRIVAL_SEEDS = tuple(range(12081, 12101))
PREVIOUSLY_USED_ARRIVAL_SEEDS = tuple(
    sorted({*raw.PREVIOUSLY_USED_ARRIVAL_SEEDS, *raw.ARRIVAL_SEEDS})
)
ARRIVAL_HORIZON_S = development.ARRIVAL_HORIZON_S
PRIMARY_BUDGETS = development.PRIMARY_BUDGETS
CONTINUATION_BUDGETS = development.CONTINUATION_BUDGETS
REFERENCE_BUDGET = development.REFERENCE_BUDGET
WORK_FIELDS = development.WORK_FIELDS
WORK_GATE_FIELDS = development.WORK_GATE_FIELDS
EXECUTION_WORKERS = development.EXECUTION_WORKERS
THIS_SOURCE = (
    "algorithm/experiments/"
    "aligned_hardware_joint95_v5_envelope_"
    "tie_continuation_confirmation.py"
)
HISTORICAL_SOURCE_PREFIX = development.HISTORICAL_SOURCE_PREFIX


class TieContinuationConfirmationRefusal(RuntimeError):
    pass


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TieContinuationConfirmationRefusal(f"expected JSON object: {path}")
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
    rows[THIS_SOURCE] = (ROOT / THIS_SOURCE).read_text(encoding="utf-8")
    return dict(sorted(rows.items()))


def _validate_seed_roster() -> None:
    old = set(PREVIOUSLY_USED_ARRIVAL_SEEDS)
    if (
        len(raw.PREVIOUSLY_USED_ARRIVAL_SEEDS) != 133
        or tuple(raw.ARRIVAL_SEEDS) != tuple(range(12061, 12081))
        or len(PREVIOUSLY_USED_ARRIVAL_SEEDS) != 153
        or ARRIVAL_SEEDS != tuple(range(12081, 12101))
        or len(set(ARRIVAL_SEEDS)) != 20
        or old.intersection(ARRIVAL_SEEDS)
    ):
        raise TieContinuationConfirmationRefusal(
            "independent arrival-seed roster is not the frozen disjoint successor"
        )


def _validated_development_bundle(
    paths: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if set(paths) != {
        "development_registration",
        "development_fixed_sources",
        "development_result",
    }:
        raise TieContinuationConfirmationRefusal("development paths changed")
    protocol = _read_object(Path(paths["development_registration"]))
    frozen = _read_object(Path(paths["development_fixed_sources"]))
    result = _read_object(Path(paths["development_result"]))
    source_paths = protocol.get("source_paths") or {}
    try:
        expected = development.build_registration(
            predecessor_registration=Path(source_paths["predecessor_registration"]),
            predecessor_fixed_sources=Path(source_paths["predecessor_fixed_sources"]),
            predecessor_result=Path(source_paths["predecessor_result"]),
        )
        development._validate_protocol(protocol, expected)
        current_development_sources = development._current_fixed_sources(
            Path(source_paths["predecessor_fixed_sources"])
        )
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        raise TieContinuationConfirmationRefusal(
            f"development lineage is invalid: {exc}"
        ) from exc

    normal_keys = sorted(
        key for key in frozen if not key.startswith(HISTORICAL_SOURCE_PREFIX)
    )
    missing = [key for key in normal_keys if not (ROOT / key).is_file()]
    drift = [
        key
        for key in normal_keys
        if (ROOT / key).is_file()
        and (ROOT / key).read_text(encoding="utf-8") != frozen[key]
    ]
    if (
        len(frozen) != 61
        or len(normal_keys) != 31
        or missing
        or drift
        or frozen != current_development_sources
        or development.THIS_SOURCE not in frozen
    ):
        raise TieContinuationConfirmationRefusal(
            "development fixed sources or current normal sources changed"
        )

    checks = result.get("checks") or {}
    cases = result.get("case_summaries") or ()
    runs = result.get("runs") or ()
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
    if (
        result.get("protocol") != protocol
        or result.get("gate") != development.GATE
        or result.get("experiment") != development.EXPERIMENT
        or result.get("status") != "PASS"
        or result.get("scientific_decision") != development.PASS_DECISION
        or result.get("development_replay_passed") is not True
        or result.get("new_seed_independent_confirmation_authorized") is not True
        or result.get("independent_confirmation_eligible") is not False
        or result.get("trajectory_or_flow_claim_authorized") is not False
        or result.get("paper_claim_authorized") is not False
        or result.get("trajectory_runs") != 240
        or result.get("arrival_generations") != 120
        or result.get("shadow_trajectory_runs") != 0
        or result.get("server_pulls") != 0
        or result.get("tie_continuation_trigger_count") != 2
        or result.get("tie_continuation_final_exact_count") != 2
        or result.get(
            "all_triggered_development_states_reached_canonical_exact"
        )
        is not True
        or not checks
        or not all(checks.values())
        or len(cases) != 6
        or not all(
            row.get("case_gate_passed") is True
            and row.get("execution_integrity_passed") is True
            and row.get("trajectory_support_gate_passed") is True
            and row.get("same_state_action_and_regret_gate_passed") is True
            and row.get("paired_mean_flow_noninferiority_passed") is True
            and row.get("case_compute_gate_passed") is True
            and row.get("seeds") == list(development.ARRIVAL_SEEDS)
            for row in cases
        )
        or observed_run_keys != expected_run_keys
        or not (result.get("work_gates") or {}).get("gate_passed")
    ):
        raise TieContinuationConfirmationRefusal(
            "development is not the required audited PASS"
        )

    contract = {
        "experiment": result["experiment"],
        "gate": result["gate"],
        "status": result["status"],
        "scientific_decision": result["scientific_decision"],
        "arrival_seeds": list(development.ARRIVAL_SEEDS),
        "arrival_generations": result["arrival_generations"],
        "trajectory_runs": result["trajectory_runs"],
        "tie_continuation_trigger_count": result[
            "tie_continuation_trigger_count"
        ],
        "tie_continuation_final_exact_count": result[
            "tie_continuation_final_exact_count"
        ],
        "checks": _json(checks),
        "case_outcomes": [
            {
                "lanes": row["lanes"],
                "load": row["load"],
                "seed_count": row["seed_count"],
                "candidate_frame_count": row["candidate_frame_count"],
                "tie_continuation_trigger_count": row[
                    "tie_continuation_trigger_count"
                ],
                "tie_continuation_final_exact_count": row[
                    "tie_continuation_final_exact_count"
                ],
                "case_gate_passed": row["case_gate_passed"],
            }
            for row in cases
        ],
        "work_gates": _json(result["work_gates"]),
        "fixed_source_keys": sorted(frozen),
        "fixed_source_count": len(frozen),
        "current_normal_source_keys": normal_keys,
        "current_normal_source_drift": [],
        "independent_confirmation_authorized": True,
    }
    capacity = protocol.get("physical_envelope_capacity_normalized_loads")
    eta = protocol.get("lane_minimum_effective_slack_eta")
    if not isinstance(capacity, dict) or not isinstance(eta, dict):
        raise TieContinuationConfirmationRefusal("development envelope is missing")
    return contract, protocol, _json(capacity), _json(eta)


TRAJECTORY_CONTRACT = {
    "lanes": list(LANES),
    "loads": list(LOADS),
    "arrival_seeds": list(ARRIVAL_SEEDS),
    "previously_used_arrival_seeds": list(PREVIOUSLY_USED_ARRIVAL_SEEDS),
    "arrival_horizon_s": ARRIVAL_HORIZON_S,
    "case_seed_tapes": len(LANES) * len(LOADS) * len(ARRIVAL_SEEDS),
    "executed_policies": ["candidate", "cold_k4096_reference"],
    "trajectory_runs": 2 * len(LANES) * len(LOADS) * len(ARRIVAL_SEEDS),
    "candidate_method": _json(development.METHOD_CONTRACT),
    "candidate_state_shadow": (
        "one independent cold K4096 call on every candidate pre-dispatch state; "
        "audit only and never read by the candidate"
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
    "rng_initializations": 0,
    "rng_draws": 0,
    "arrival_generations": 0,
    "selector_calls": 0,
    "trajectory_runs": 0,
    "server_pulls": 0,
    "paper_or_readme_edits": 0,
}


CLAIM_BOUNDARY = (
    "A PASS is independent confirmation of the fixed score-exact tie-continuation "
    "method only for this preregistered model, R8/R16 lane counts, 0.5/0.8/0.95 "
    "normalized loads, 12081--12100 synthetic Poisson tapes, 600-second arrival "
    "window plus full drain, FIFO execution, K4096 same-state audit, paired flow "
    "endpoint, and registered compute gates. It does not authorize a paper edit or "
    "a broader stability, capacity, physical-hardware, or all-queue claim."
)


def build_registration(**kwargs: Any) -> dict[str, Any]:
    _validate_seed_roster()
    paths = _source_paths(**kwargs)
    contract, source_protocol, capacity, eta = _validated_development_bundle(paths)
    model = source_protocol.get("physical_complete_frame_model") or {}
    if (
        len(model.get("options") or ()) != 9
        or source_protocol.get("arrival_seeds")
        != list(development.ARRIVAL_SEEDS)
        or source_protocol.get("arrival_horizon_s") != ARRIVAL_HORIZON_S
        or source_protocol.get("method_contract") != development.METHOD_CONTRACT
        or set(capacity) != {"8", "16"}
        or set(eta) != {"8", "16"}
    ):
        raise TieContinuationConfirmationRefusal(
            "development model, method or envelope changed"
        )
    return {
        "schema_version": 1,
        "gate": GATE,
        "experiment": EXPERIMENT,
        "status": REGISTRATION_STATUS,
        "registered": True,
        "independent_confirmation": True,
        "source_paths": paths,
        "development_pass_contract": contract,
        "physical_complete_frame_model": _json(model),
        "physical_envelope_capacity_normalized_loads": capacity,
        "lane_minimum_effective_slack_eta": eta,
        "trajectory_contract": _json(TRAJECTORY_CONTRACT),
        "work_fields": list(WORK_FIELDS),
        "work_gate_fields": list(WORK_GATE_FIELDS),
        "scientific_gates": {
            "execution_completeness_certificate_and_fifo": (
                "every registered execution-integrity check must pass"
            ),
            "support": "eta exceeds maximum candidate epsilon and actual regret",
            "action_and_regret": (
                "every candidate action matches its exact same-state K4096 shadow "
                "and actual regret is bounded by the certificate"
            ),
            "flow": "six-family paired mean-flow upper bounds are at most 1.01",
            "case_compute": (
                "candidate <= same-state K4096 shadow in every gate field"
            ),
            "lane_compute": (
                "candidate < same-state K4096 shadow in every gate field"
            ),
            "aggregate_compute": (
                "candidate < same-state K4096 shadow in every gate field"
            ),
        },
        "registration_scope": dict(REGISTRATION_SCOPE),
        "claim_boundary": CLAIM_BOUNDARY,
        "paper_claim_authorized": False,
    }


def _validate_protocol(protocol: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    observed = dict(protocol)
    registered_at = observed.pop("registered_at_utc", None)
    if not isinstance(registered_at, str) or observed != expected:
        raise TieContinuationConfirmationRefusal("registered protocol changed")


def register(
    registration: Path = DEFAULT_REGISTRATION,
    fixed_sources: Path = DEFAULT_FIXED_SOURCES,
    result: Path = DEFAULT_RESULT,
    **kwargs: Any,
) -> dict[str, Any]:
    if any(Path(path).exists() for path in (registration, fixed_sources, result)):
        raise TieContinuationConfirmationRefusal("confirmation artifact already exists")
    body = build_registration(**kwargs)
    protocol = {"registered_at_utc": datetime.now(timezone.utc).isoformat(), **body}
    source_fixed = Path(body["source_paths"]["development_fixed_sources"])
    _write_new(Path(fixed_sources), _current_fixed_sources(source_fixed))
    _write_new(Path(registration), protocol, compact=True)
    return protocol


def _paired_case_summary(
    runs: Sequence[Mapping[str, Any]],
    lanes: int,
    load: float,
    eta: float,
    expected_seeds: Sequence[int],
) -> dict[str, Any]:
    seed_roster = tuple(int(seed) for seed in expected_seeds)
    members = sorted(
        (row for row in runs if row["lanes"] == lanes and row["load"] == load),
        key=lambda row: row["seed"],
    )
    if [row["seed"] for row in members] != list(seed_roster):
        raise TieContinuationConfirmationRefusal(
            "case does not contain the independent registered seed roster"
        )
    ratios: list[float] = []
    seed_rows = []
    jobs_complete = certificates_valid = flow_identities = True
    shadows_exact = regret_bounded = action_identity = recursion_valid = True
    max_epsilon = max_regret = 0.0
    candidate_frames = candidate_exact_frames = 0
    for row in members:
        policies = {item["policy"]: item for item in row["policies"]}
        candidate = policies["candidate"]
        reference = policies["cold_k4096_reference"]
        cm, rm = candidate["metrics"], reference["metrics"]
        if cm["mean_flow_s"] is None or rm["mean_flow_s"] in (None, 0):
            raise TieContinuationConfirmationRefusal(
                "paired mean-flow ratio is undefined"
            )
        ratio = cm["mean_flow_s"] / rm["mean_flow_s"]
        ratios.append(ratio)
        max_epsilon = max(max_epsilon, cm["maximum_certified_gap_per_job"])
        max_regret = max(max_regret, cm["maximum_actual_normalized_regret"])
        candidate_frames += cm["frame_count"]
        candidate_exact_frames += cm["exact_selection_frame_count"]
        jobs_complete &= bool(
            cm["job_count"]
            == cm["completed_jobs"]
            == rm["job_count"]
            == rm["completed_jobs"]
            == row["arrival_tape"]["job_count"]
        )
        certificates_valid &= bool(
            cm["valid_selection_frame_count"] == cm["frame_count"]
            and rm["valid_selection_frame_count"] == rm["frame_count"]
            and rm["exact_selection_frame_count"] == rm["frame_count"]
        )
        shadows_exact &= bool(
            cm["same_state_shadow_valid_count"] == cm["frame_count"]
            and cm["same_state_shadow_exact_count"] == cm["frame_count"]
        )
        regret_bounded &= cm["regret_bounded_by_epsilon_count"] == cm["frame_count"]
        action_identity &= cm["same_state_action_match_count"] == cm["frame_count"]
        recursion_valid &= bool(
            cm["maximum_queue_recursion_violation"] <= raw.EXECUTION_TOLERANCE
            and rm["maximum_queue_recursion_violation"] <= raw.EXECUTION_TOLERANCE
        )
        flow_identities &= all(
            item["flow_decomposition"]["status"] == "PASS"
            and abs(
                item["flow_decomposition"]["mean_flow_s"]
                - item["metrics"]["mean_flow_s"]
            )
            <= raw.EXECUTION_TOLERANCE
            for item in policies.values()
        )
        seed_rows.append(
            {
                "seed": row["seed"],
                "job_count": cm["job_count"],
                "candidate_mean_flow_s": cm["mean_flow_s"],
                "reference_mean_flow_s": rm["mean_flow_s"],
                "candidate_over_reference_mean_flow_ratio": ratio,
                "candidate_frame_count": cm["frame_count"],
                "candidate_inexact_frame_count": cm["inexact_selection_frame_count"],
                "reference_frame_count": rm["frame_count"],
                "candidate_maximum_epsilon": cm[
                    "maximum_certified_gap_per_job"
                ],
                "candidate_maximum_actual_regret": cm[
                    "maximum_actual_normalized_regret"
                ],
            }
        )
    estimate = mean(ratios)
    standard_error = stdev(ratios) / math.sqrt(len(ratios))
    per_case_alpha = raw.FAMILY_ALPHA / raw.PRIMARY_FAMILY_SIZE
    critical = float(student_t.ppf(1.0 - per_case_alpha, df=len(ratios) - 1))
    upper = estimate + critical * standard_error
    flow_pass = bool(
        upper
        <= 1.0 + raw.NONINFERIORITY_MARGIN + development.SCORE_TOLERANCE
    )
    support_pass = bool(
        eta - max_epsilon > development.SCORE_TOLERANCE
        and eta - max_regret > development.SCORE_TOLERANCE
    )
    action_pass = bool(action_identity and regret_bounded)
    case_candidate, case_shadow = raw._sum_run_work(members)
    comparison = raw._work_comparison(case_candidate, case_shadow)
    compute_pass = comparison["gate_fields_no_greater"]
    execution_pass = bool(
        jobs_complete
        and certificates_valid
        and flow_identities
        and shadows_exact
        and recursion_valid
    )
    return {
        "lanes": lanes,
        "load": load,
        "candidate_fixed_budget": raw.CANDIDATE_BUDGETS[lanes],
        "cold_reference_budget": REFERENCE_BUDGET,
        "seeds": list(seed_roster),
        "seed_count": len(members),
        "paired_seed_rows": seed_rows,
        "paired_mean_flow_ratios": ratios,
        "paired_ratio_mean": estimate,
        "paired_ratio_standard_error": standard_error,
        "one_sided_confidence_level": 1.0 - per_case_alpha,
        "critical_t": critical,
        "paired_ratio_upper_confidence_bound": upper,
        "noninferiority_acceptance_upper": 1.0 + raw.NONINFERIORITY_MARGIN,
        "paired_mean_flow_noninferiority_passed": flow_pass,
        "effective_slack_eta": eta,
        "maximum_candidate_epsilon": max_epsilon,
        "maximum_actual_normalized_regret": max_regret,
        "eta_minus_candidate_epsilon": eta - max_epsilon,
        "eta_minus_actual_regret": eta - max_regret,
        "trajectory_support_gate_passed": support_pass,
        "same_state_action_and_regret_gate_passed": action_pass,
        "candidate_frame_count": candidate_frames,
        "candidate_exact_frame_count": candidate_exact_frames,
        "candidate_inexact_frame_count": candidate_frames - candidate_exact_frames,
        "candidate_exactness_is_diagnostic_not_a_gate": True,
        "charged_candidate_work": case_candidate,
        "same_state_cold_k4096_shadow_work": case_shadow,
        "charged_work_comparison": comparison,
        "case_compute_rule": "all_three_gate_fields_no_greater",
        "case_compute_gate_passed": compute_pass,
        "execution_integrity_passed": execution_pass,
        "case_gate_passed": bool(
            execution_pass and support_pass and action_pass and flow_pass and compute_pass
        ),
    }


def _case_summary(
    runs: Sequence[Mapping[str, Any]], lanes: int, load: float, eta: float
) -> dict[str, Any]:
    summary = _paired_case_summary(runs, lanes, load, eta, ARRIVAL_SEEDS)
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


def _execute_tasks(
    tasks: Sequence[tuple[Any, Mapping[str, Any], int, float, int, float]],
) -> list[dict[str, Any]]:
    with ProcessPoolExecutor(max_workers=EXECUTION_WORKERS) as pool:
        return list(pool.map(development._run_tape, tasks))


def run_registered(
    registration: Path = DEFAULT_REGISTRATION,
    fixed_sources: Path = DEFAULT_FIXED_SOURCES,
    output: Path = DEFAULT_RESULT,
) -> dict[str, Any]:
    if Path(output).exists():
        raise TieContinuationConfirmationRefusal(f"result exists: {output}")
    protocol = _read_object(Path(registration))
    paths = protocol.get("source_paths") or {}
    source_contract, _, _, _ = _validated_development_bundle(paths)
    expected = build_registration(
        development_registration=Path(paths["development_registration"]),
        development_fixed_sources=Path(paths["development_fixed_sources"]),
        development_result=Path(paths["development_result"]),
    )
    _validate_protocol(protocol, expected)
    if source_contract != protocol["development_pass_contract"]:
        raise TieContinuationConfirmationRefusal("development PASS contract changed")
    registered_sources = _read_object(Path(fixed_sources))
    current_sources = _current_fixed_sources(
        Path(paths["development_fixed_sources"])
    )
    if registered_sources != current_sources:
        raise TieContinuationConfirmationRefusal("registered sources changed")

    model = raw.old_transition.baseline._model_from_snapshot(
        protocol["physical_complete_frame_model"]
    )
    if _json(model.snapshot()) != protocol["physical_complete_frame_model"]:
        raise TieContinuationConfirmationRefusal("frozen model reconstruction failed")
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
                raise TieContinuationConfirmationRefusal("capacity row changed")
            by_case[key] = row
    expected_cases = {(lanes, load) for lanes in LANES for load in LOADS}
    if set(by_case) != expected_cases:
        raise TieContinuationConfirmationRefusal("capacity grid changed")
    tasks = [
        (model, by_case[(lanes, load)], lanes, load, seed, ARRIVAL_HORIZON_S)
        for lanes in LANES
        for load in LOADS
        for seed in ARRIVAL_SEEDS
    ]
    started = time.perf_counter()
    runs = _execute_tasks(tasks)

    post_expected = build_registration(
        development_registration=Path(paths["development_registration"]),
        development_fixed_sources=Path(paths["development_fixed_sources"]),
        development_result=Path(paths["development_result"]),
    )
    _validate_protocol(protocol, post_expected)
    if (
        _read_object(Path(fixed_sources)) != registered_sources
        or registered_sources
        != _current_fixed_sources(Path(paths["development_fixed_sources"]))
    ):
        raise TieContinuationConfirmationRefusal(
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
        "development_pass_and_61_sources_revalidated": True,
        "development_current_normal_sources_have_zero_drift": (
            source_contract["current_normal_source_drift"] == []
        ),
        "protocol_and_62_sources_revalidated_after_execution": True,
        "independent_seed_roster_is_disjoint_from_all_153_prior_seeds": not (
            set(ARRIVAL_SEEDS).intersection(PREVIOUSLY_USED_ARRIVAL_SEEDS)
        ),
        "all_120_registered_case_seed_tapes_execute_exactly_once": (
            observed_run_keys == expected_run_keys and len(runs) == 120
        ),
        "executed_trajectory_count_is_240": len(policy_rows) == 240,
        "each_tape_executes_candidate_and_reference_once": all(
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
        "one_conditional_selector_call_per_candidate_frame": all(
            row["conditional_selector_call_count"] == row["frame_count"]
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
        "every_candidate_frame_has_one_fresh_valid_exact_k4096_shadow": all(
            row["same_state_shadow_calls"]
            == row["same_state_shadow_valid_count"]
            == row["same_state_shadow_exact_count"]
            == row["frame_count"]
            for row in candidate_metrics
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
        "tie_continuation_final_exact_count": sum(
            row["tie_continuation_final_exact_count"] for row in candidate_metrics
        ),
        "all_triggered_confirmation_states_reached_canonical_exact": trigger_count
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
        "independent_confirmation_passed": passed,
        "trajectory_transition_passed": passed,
        "trajectory_or_flow_claim_authorized": passed,
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
    except TieContinuationConfirmationRefusal as exc:
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
