"""New-seed trajectory transition for the post-scout raw fixed-cap selector.

Registration is allowed only after the statewise cumulative-continuation
development selects R8/K288 and R16/K576.  The runner then executes the raw
candidate and a cold K4096 reference on common fresh arrival tapes.  Every
candidate state also receives a separate nonexecuted cold K4096 shadow.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import mean, stdev
import time
from typing import Any, Mapping, Sequence

from scipy.stats import t as student_t

from algorithm.canonical_count_space_bnb_frame_candidates import (
    select_canonical_count_space_bnb_frame,
)
from algorithm.experiments import (
    aligned_hardware_count_space_bnb_canonical_exact_compute_confirmation
    as seed_history,
)
from algorithm.experiments import (
    aligned_hardware_joint95_v5_envelope_cumulative_continuation_development
    as development,
)
from algorithm.experiments import (
    aligned_hardware_joint95_v5_envelope_normalized_trajectory_transition
    as old_transition,
)
from algorithm.experiments.aligned_frame_flow_decomposition import decompose_flow
from algorithm.experiments.aligned_hardware_frame_experiment import poisson_arrivals


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
STEM = (
    "aligned_hardware_joint95_v5_envelope_"
    "raw_fixed_cap_trajectory_transition_20260920"
)
DEFAULT_REGISTRATION = ARTIFACTS / f"{STEM}_protocol.json"
DEFAULT_FIXED_SOURCES = ARTIFACTS / f"{STEM}_fixed_sources.json"
DEFAULT_RESULT = ARTIFACTS / f"{STEM}.json"
DEFAULT_DEVELOPMENT_REGISTRATION = development.DEFAULT_REGISTRATION
DEFAULT_DEVELOPMENT_FIXED_SOURCES = development.DEFAULT_FIXED_SOURCES
DEFAULT_DEVELOPMENT_RESULT = development.DEFAULT_RESULT

EXPERIMENT = (
    "aligned_hardware_joint95_v5_envelope_"
    "raw_fixed_cap_trajectory_transition_v1"
)
GATE = (
    "aligned_hardware_joint95_v5_envelope_"
    "raw_fixed_cap_trajectory_transition_runner"
)
REGISTRATION_STATUS = "REGISTERED_AWAITING_RAW_FIXED_CAP_NEW_SEED_TRAJECTORY"
PASS_DECISION = "RAW_FIXED_CAP_NEW_SEED_TRAJECTORY_PASS"
INVALID_DECISION = "INVALID_RAW_FIXED_CAP_TRAJECTORY_EXECUTION"
SUPPORT_FAIL_DECISION = "RAW_FIXED_CAP_TRAJECTORY_SUPPORT_FAIL_STOP"
ACTION_FAIL_DECISION = "RAW_FIXED_CAP_TRAJECTORY_ACTION_OR_REGRET_FAIL_STOP"
COMPUTE_FAIL_DECISION = "RAW_FIXED_CAP_TRAJECTORY_COMPUTE_FAIL_STOP"
FLOW_FAIL_DECISION = "RAW_FIXED_CAP_TRAJECTORY_FLOW_NONINFERIORITY_FAIL_STOP"

LANES = (8, 16)
LOADS = (0.5, 0.8, 0.95)
CANDIDATE_BUDGETS = {8: 288, 16: 576}
REFERENCE_BUDGET = 4096
ARRIVAL_SEEDS = tuple(range(12061, 12081))
PREVIOUSLY_USED_ARRIVAL_SEEDS = tuple(sorted({
    *seed_history.PREVIOUSLY_USED_ARRIVAL_SEEDS,
    *seed_history.ARRIVAL_SEEDS,
    *old_transition.ARRIVAL_SEEDS,
}))
STATIC_ROSTER_RNG_SEED = 10041
ARRIVAL_HORIZON_S = 600.0
NONINFERIORITY_MARGIN = 0.01
FAMILY_ALPHA = 0.05
PRIMARY_FAMILY_SIZE = len(LANES) * len(LOADS)
EXECUTION_WORKERS = 4
EXECUTION_TOLERANCE = 1e-8
WORK_FIELDS = development.WORK_FIELDS
WORK_GATE_FIELDS = development.WORK_GATE_FIELDS
SCORE_TOLERANCE = development.SCORE_TOLERANCE
THIS_SOURCE = (
    "algorithm/experiments/"
    "aligned_hardware_joint95_v5_envelope_"
    "raw_fixed_cap_trajectory_transition.py"
)


class RawFixedCapTrajectoryRefusal(RuntimeError):
    pass


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RawFixedCapTrajectoryRefusal(f"expected JSON object: {path}")
    return payload


def _write_new(path: Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=None if compact else 2, sort_keys=True)
        handle.write("\n")


def _new_work() -> dict[str, int]:
    return {field: 0 for field in WORK_FIELDS}


def _checked_work(work: Mapping[str, Any]) -> dict[str, int]:
    if set(work) != set(WORK_FIELDS):
        raise RawFixedCapTrajectoryRefusal("work row does not cover WORK_FIELDS")
    row = {field: int(work[field]) for field in WORK_FIELDS}
    if any(row[field] < 0 or row[field] != work[field] for field in WORK_FIELDS):
        raise RawFixedCapTrajectoryRefusal("invalid work count")
    return row


def _add_work(target: dict[str, int], row: Mapping[str, Any]) -> None:
    checked = _checked_work(row)
    for field in WORK_FIELDS:
        target[field] += checked[field]


def _selection_work(selection: Any) -> dict[str, int]:
    return _checked_work(
        {field: int(getattr(selection, field)) for field in WORK_FIELDS}
    )


def _current_fixed_sources() -> dict[str, str]:
    rows = _read_object(DEFAULT_DEVELOPMENT_FIXED_SOURCES)
    rows[THIS_SOURCE] = (ROOT / THIS_SOURCE).read_text(encoding="utf-8")
    return dict(sorted(rows.items()))


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


def _validated_development_bundle(
    paths: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = _read_object(Path(paths["development_registration"]))
    frozen = _read_object(Path(paths["development_fixed_sources"]))
    result = _read_object(Path(paths["development_result"]))
    source_paths = protocol.get("source_paths") or {}
    expected = development.build_registration(
        source_registration=Path(source_paths["frontier_registration"]),
        source_fixed_sources=Path(source_paths["frontier_fixed_sources"]),
        source_result=Path(source_paths["frontier_result"]),
    )
    development._validate_protocol(protocol, expected)
    if frozen != development._current_fixed_sources():
        raise RawFixedCapTrajectoryRefusal("development fixed sources changed")
    selected = result.get("selected_lane_caps") or {}
    if (
        result.get("protocol") != protocol
        or result.get("gate") != development.GATE
        or result.get("experiment") != development.EXPERIMENT
        or result.get("status") != "PASS"
        or result.get("scientific_decision") != development.PASS_DECISION
        or selected != {"8": 288, "16": 576}
        or result.get(
            "implementation_and_new_seed_trajectory_confirmation_authorized"
        )
        is not True
        or result.get("trajectory_or_flow_claim_authorized") is not False
        or result.get("paper_claim_authorized") is not False
        or result.get("trajectory_runs") != 0
        or result.get("arrival_generations") != 0
        or result.get("server_pulls") != 0
        or not result.get("checks")
        or not all(result["checks"].values())
    ):
        raise RawFixedCapTrajectoryRefusal(
            "statewise raw fixed-cap development is not the required PASS"
        )
    frontier_protocol = _read_object(Path(source_paths["frontier_registration"]))
    capacity = frontier_protocol.get("physical_envelope_capacity_normalized_loads")
    if not isinstance(capacity, dict):
        raise RawFixedCapTrajectoryRefusal("frozen normalized capacity rows are missing")
    contract = {
        "experiment": result["experiment"],
        "gate": result["gate"],
        "status": result["status"],
        "scientific_decision": result["scientific_decision"],
        "selected_lane_caps": _json(selected),
        "checks": _json(result["checks"]),
        "unique_state_count": result["unique_state_count"],
        "state_occurrence_count": result["state_occurrence_count"],
        "implementation_and_new_seed_trajectory_confirmation_authorized": True,
        "source_fixed_source_keys": sorted(frozen),
    }
    return contract, protocol, _json(capacity)


TRAJECTORY_CONTRACT = {
    "lanes": list(LANES),
    "loads": list(LOADS),
    "arrival_seeds": list(ARRIVAL_SEEDS),
    "previously_used_arrival_seeds": list(PREVIOUSLY_USED_ARRIVAL_SEEDS),
    "static_roster_rng_seed_avoided": STATIC_ROSTER_RNG_SEED,
    "arrival_horizon_s": ARRIVAL_HORIZON_S,
    "case_seed_tapes": len(LANES) * len(LOADS) * len(ARRIVAL_SEEDS),
    "executed_policies": ["candidate", "cold_k4096_reference"],
    "trajectory_runs": 2 * len(LANES) * len(LOADS) * len(ARRIVAL_SEEDS),
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
        "selectors receive only the released current queue, model, lanes and fixed budget"
    ),
    "stopping_rule": "stop arrivals at 600 seconds and fully drain both arms",
    "candidate_fixed_budget_by_lane": {
        str(lanes): CANDIDATE_BUDGETS[lanes] for lanes in LANES
    },
    "candidate_method": (
        "one cold native canonical count-space B&B call; no trace, trigger, "
        "single-exchange, bulk polish, or rescue"
    ),
    "reference_budget": REFERENCE_BUDGET,
    "paired_endpoint": "candidate/reference arithmetic mean-flow ratio",
    "noninferiority_margin": NONINFERIORITY_MARGIN,
    "family_alpha": FAMILY_ALPHA,
    "primary_family_size": PRIMARY_FAMILY_SIZE,
}


REGISTRATION_SCOPE = {
    "post_scout_candidate_and_compute_gate": True,
    "independent_new_seed_trajectory": True,
    "development_result_read": True,
    "rng_initializations": 0,
    "rng_draws": 0,
    "arrival_generations": 0,
    "selector_calls": 0,
    "trajectory_runs": 0,
    "capacity_recomputations": 0,
    "server_pulls": 0,
    "paper_or_readme_edits": 0,
}


CLAIM_BOUNDARY = (
    "A fixed-method new-seed 600-second plus full-drain FIFO trajectory "
    "transition on 120 Poisson tapes. Caps and the uniform case/lane/aggregate "
    "compute rule were selected after the earlier development tapes. This run "
    "tests those choices on disjoint arrival seeds; it does not establish a "
    "stability theorem, capacity closure, physical multi-GPU performance, or a "
    "paper claim."
)


def build_registration(**kwargs: Any) -> dict[str, Any]:
    paths = _source_paths(**kwargs)
    contract, source_protocol, capacity = _validated_development_bundle(paths)
    if set(ARRIVAL_SEEDS) & set(PREVIOUSLY_USED_ARRIVAL_SEEDS):
        raise RawFixedCapTrajectoryRefusal("new arrival seeds overlap prior tapes")
    if STATIC_ROSTER_RNG_SEED in ARRIVAL_SEEDS:
        raise RawFixedCapTrajectoryRefusal("arrival seeds reuse the static roster seed")
    if len(PREVIOUSLY_USED_ARRIVAL_SEEDS) != 133:
        raise RawFixedCapTrajectoryRefusal("frozen prior arrival-seed union changed")
    model = source_protocol["physical_complete_frame_model"]
    eta = source_protocol["lane_minimum_effective_slack_eta"]
    if (
        len(model.get("options") or ()) != 9
        or set(capacity) != {"8", "16"}
        or set(eta) != {"8", "16"}
        or any(
            [float(row["load"]) for row in capacity[str(lanes)]] != list(LOADS)
            for lanes in LANES
        )
    ):
        raise RawFixedCapTrajectoryRefusal("frozen model, eta or capacity grid changed")
    return {
        "schema_version": 1,
        "gate": GATE,
        "experiment": EXPERIMENT,
        "status": REGISTRATION_STATUS,
        "registered": True,
        "source_paths": paths,
        "development_pass_contract": contract,
        "physical_complete_frame_model": _json(model),
        "physical_envelope_capacity_normalized_loads": capacity,
        "lane_minimum_effective_slack_eta": _json(eta),
        "trajectory_contract": _json(TRAJECTORY_CONTRACT),
        "work_fields": list(WORK_FIELDS),
        "work_gate_fields": list(WORK_GATE_FIELDS),
        "compute_gate": {
            "case": (
                "candidate <= same-state K4096 shadow in every gate field, "
                "uniformly for all six cases without exceptions"
            ),
            "lane": "candidate < same-state K4096 shadow in every gate field",
            "aggregate": "candidate < same-state K4096 shadow in every gate field",
            "integer_comparison_has_no_tolerance": True,
            "no_cross_field_tradeoff": True,
        },
        "registration_scope": dict(REGISTRATION_SCOPE),
        "claim_boundary": CLAIM_BOUNDARY,
        "paper_claim_authorized": False,
    }


def _validate_protocol(protocol: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    observed = dict(protocol)
    registered_at = observed.pop("registered_at_utc", None)
    if not isinstance(registered_at, str) or observed != expected:
        raise RawFixedCapTrajectoryRefusal("registered trajectory protocol changed")


def register(
    registration: Path = DEFAULT_REGISTRATION,
    fixed_sources: Path = DEFAULT_FIXED_SOURCES,
    result: Path = DEFAULT_RESULT,
    **kwargs: Any,
) -> dict[str, Any]:
    if any(Path(path).exists() for path in (registration, fixed_sources, result)):
        raise RawFixedCapTrajectoryRefusal("trajectory artifact already exists")
    body = build_registration(**kwargs)
    protocol = {"registered_at_utc": datetime.now(timezone.utc).isoformat(), **body}
    _write_new(Path(fixed_sources), _current_fixed_sources())
    _write_new(Path(registration), protocol, compact=True)
    return protocol


def _certificate_valid(
    model: Any,
    queue: Sequence[int],
    budget: int,
    selection: Any,
    *,
    exact: bool = False,
) -> bool:
    valid = bool(
        old_transition.confirmation.canonical._selection_reconstructs(
            model, queue, model.lanes, selection
        )
        and old_transition.confirmation.canonical._selection_work_holds(
            queue, model.lanes, budget, selection
        )
        and old_transition.confirmation.canonical._certificate_valid(selection, queue)
    )
    if exact:
        valid = bool(
            valid
            and selection.score_exact
            and selection.canonical_exact
            and math.isclose(
                selection.score,
                selection.upper_bound,
                rel_tol=0.0,
                abs_tol=SCORE_TOLERANCE,
            )
        )
    return valid


def _candidate_call(model: Any, queue: Sequence[int]) -> dict[str, Any]:
    budget = CANDIDATE_BUDGETS[int(model.lanes)]
    selected = select_canonical_count_space_bnb_frame(
        model.options, queue, model.lanes, budget
    )
    return {
        "selection": selected,
        "work": _selection_work(selected),
        "valid": _certificate_valid(model, queue, budget, selected),
        "budget": budget,
    }


def _reference_call(model: Any, queue: Sequence[int]) -> dict[str, Any]:
    selected = select_canonical_count_space_bnb_frame(
        model.options, queue, model.lanes, REFERENCE_BUDGET
    )
    return {
        "selection": selected,
        "work": _selection_work(selected),
        "valid": _certificate_valid(
            model, queue, REFERENCE_BUDGET, selected, exact=True
        ),
    }


def _compact_decomposition(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": row["status"],
        "job_count": row["job_count"],
        "completed_jobs": row["completed_jobs"],
        "mean_flow_s": row["mean_flow_s"],
        "flow_area_job_seconds": row["flow_area_job_seconds"],
        "flow_identity_error_job_seconds": row["flow_identity_error_job_seconds"],
        "mean_deferred_visible_wait_s": row["mean_deferred_visible_wait_s"],
        "mean_arrival_barrier_wait_s": row["mean_arrival_barrier_wait_s"],
        "mean_dispatch_to_completion_s": row["mean_dispatch_to_completion_s"],
        "arrival_window_mean_flow_contribution_s": row["time_regions"][
            "arrival_window"
        ]["mean_flow_contribution_s"],
        "drain_mean_flow_contribution_s": row["time_regions"]["drain"][
            "mean_flow_contribution_s"
        ],
    }


def simulate_trajectory(
    model: Any,
    arrivals: Sequence[tuple[float, int, int]],
    horizon: float,
    policy: str,
    *,
    audit_limit: int = 3,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if policy not in {"candidate", "cold_k4096_reference"}:
        raise ValueError("unknown trajectory policy")
    queues = [deque() for _ in model.workloads]
    tape = tuple(sorted(arrivals))
    pending = iter(tape)
    next_arrival = next(pending, None)
    now = 0.0
    frames: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    flows: list[float] = []
    last_completion = 0.0
    peak_backlog = backlog_sum = recursion_violation = 0
    valid_frames = exact_frames = 0
    shadow_valid_frames = shadow_exact_frames = 0
    action_matches = bounded_regrets = 0
    max_epsilon = max_regret = max_gap = 0.0
    work = _new_work()
    shadow_work = _new_work()
    selected_counts: Counter[str] = Counter()
    selector_wall = shadow_wall = 0.0
    first_mismatch = first_regret_violation = None

    while now < horizon or next_arrival is not None or any(queues):
        while next_arrival is not None and next_arrival[0] <= now:
            at, coordinate, index = next_arrival
            queues[coordinate].append((at, index))
            next_arrival = next(pending, None)
        before = tuple(len(row) for row in queues)
        started = time.perf_counter()
        if policy == "candidate":
            call = _candidate_call(model, before)
        else:
            call = _reference_call(model, before)
        selector_wall += time.perf_counter() - started
        selected = call["selection"]
        _add_work(work, call["work"])
        valid = bool(call["valid"])

        if policy == "candidate":
            shadow_started = time.perf_counter()
            shadow_call = _reference_call(model, before)
            shadow_wall += time.perf_counter() - shadow_started
            shadow = shadow_call["selection"]
            _add_work(shadow_work, shadow_call["work"])
            shadow_valid = bool(shadow_call["valid"])
            shadow_valid_frames += int(shadow_valid)
            shadow_exact_frames += int(
                shadow.score_exact and shadow.canonical_exact
            )
            valid = bool(valid and shadow_valid)
            normalizer = sum(before)
            regret = (
                (shadow.score - selected.score) / normalizer
                if normalizer
                else 0.0
            )
            bounded = bool(
                -SCORE_TOLERANCE
                <= regret
                <= selected.epsilon + SCORE_TOLERANCE
            )
            matched = selected.option_indices == shadow.option_indices
            action_matches += int(matched)
            bounded_regrets += int(bounded)
            max_regret = max(max_regret, regret)
            if not matched and first_mismatch is None:
                first_mismatch = {
                    "frame": len(frames),
                    "queue": list(before),
                    "candidate": list(selected.option_indices),
                    "shadow": list(shadow.option_indices),
                    "regret": regret,
                }
            if not bounded and first_regret_violation is None:
                first_regret_violation = {
                    "frame": len(frames),
                    "queue": list(before),
                    "epsilon": selected.epsilon,
                    "regret": regret,
                }
        else:
            shadow = None
            regret = 0.0
            matched = bounded = True

        valid_frames += int(valid)
        exact_frames += int(selected.score_exact and selected.canonical_exact)
        max_epsilon = max(max_epsilon, selected.epsilon)
        max_gap = max(max_gap, selected.gap)
        peak_backlog = max(peak_backlog, sum(before))
        backlog_sum += sum(before)
        end = now + selected.duration
        departures = [0] * len(queues)
        for option_index in selected.option_indices:
            if option_index is None:
                selected_counts["idle"] += 1
                continue
            option = model.options[option_index]
            selected_counts[option.action_id] += 1
            for coordinate, offsets in enumerate(option.completion_offsets):
                for offset in offsets:
                    arrival_time, _ = queues[coordinate].popleft()
                    completion = now + offset
                    flows.append(completion - arrival_time)
                    last_completion = max(last_completion, completion)
                    departures[coordinate] += 1
        if tuple(departures) != selected.service:
            raise RawFixedCapTrajectoryRefusal(
                "selected service differs from FIFO departures"
            )
        during = [0] * len(queues)
        while next_arrival is not None and next_arrival[0] <= end:
            at, coordinate, index = next_arrival
            queues[coordinate].append((at, index))
            during[coordinate] += 1
            next_arrival = next(pending, None)
        after = tuple(len(row) for row in queues)
        recursion_violation = max(
            recursion_violation,
            max(
                abs(q1 - (q0 - done + added))
                for q0, done, added, q1 in zip(
                    before, departures, during, after, strict=True
                )
            ),
        )
        frame = {
            "start_s": now,
            "duration_s": selected.duration,
            "queue_before": before,
            "departures": tuple(departures),
            "option_indices": selected.option_indices,
        }
        frames.append(frame)
        if len(audits) < audit_limit:
            audits.append(
                {
                    **frame,
                    "arrivals_during_frame": tuple(during),
                    "queue_after": after,
                    "score": selected.score,
                    "upper_bound": selected.upper_bound,
                    "epsilon": selected.epsilon,
                    "score_exact": selected.score_exact,
                    "canonical_exact": selected.canonical_exact,
                    "same_state_shadow_score": shadow.score if shadow else None,
                    "same_state_shadow_action_match": matched if shadow else None,
                    "actual_normalized_regret": regret if shadow else None,
                }
            )
        now = end

    ordered = sorted(flows)
    frame_count = len(frames)
    metrics = {
        "policy": policy,
        "arrival_horizon_s": horizon,
        "elapsed_including_drain_s": now,
        "job_count": len(tape),
        "completed_jobs": len(flows),
        "makespan_s": last_completion,
        "mean_flow_s": mean(flows) if flows else None,
        "p90_flow_s": (
            ordered[math.ceil(0.9 * len(ordered)) - 1] if ordered else None
        ),
        "frame_count": frame_count,
        "valid_selection_frame_count": valid_frames,
        "exact_selection_frame_count": exact_frames,
        "inexact_selection_frame_count": frame_count - exact_frames,
        "same_state_shadow_calls": frame_count if policy == "candidate" else 0,
        "same_state_shadow_valid_count": (
            shadow_valid_frames if policy == "candidate" else None
        ),
        "same_state_shadow_exact_count": (
            shadow_exact_frames if policy == "candidate" else None
        ),
        "same_state_action_match_count": (
            action_matches if policy == "candidate" else None
        ),
        "regret_bounded_by_epsilon_count": (
            bounded_regrets if policy == "candidate" else None
        ),
        "maximum_certified_gap_per_job": max_epsilon,
        "maximum_actual_normalized_regret": (
            max_regret if policy == "candidate" else None
        ),
        "maximum_certified_gap": max_gap,
        "maximum_queue_recursion_violation": recursion_violation,
        "frame_boundary_peak_backlog": peak_backlog,
        "frame_boundary_mean_backlog": backlog_sum / frame_count,
        "work": work,
        "same_state_shadow_work": shadow_work if policy == "candidate" else None,
        "first_action_mismatch": first_mismatch,
        "first_regret_bound_violation": first_regret_violation,
        "selected_option_counts": dict(selected_counts),
        "selector_wall_s": selector_wall,
        "shadow_selector_wall_s": shadow_wall,
        "frame_audit_prefix": audits,
    }
    return metrics, frames


def _run_tape(
    task: tuple[Any, Mapping[str, Any], int, float, int, float]
) -> dict[str, Any]:
    model, capacity, lanes, load, seed, horizon = task
    lane_model = replace(model, lanes=lanes)
    tape = tuple(poisson_arrivals(capacity["arrival_rates"], horizon, seed))
    policies = []
    for policy in ("candidate", "cold_k4096_reference"):
        metrics, frames = simulate_trajectory(lane_model, tape, horizon, policy)
        decomposition = decompose_flow(lane_model, tape, frames, horizon)
        policies.append(
            {
                "policy": policy,
                "metrics": metrics,
                "flow_decomposition": _compact_decomposition(decomposition),
            }
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
        "policies": policies,
    }


def _work_comparison(
    candidate: Mapping[str, int], shadow: Mapping[str, int]
) -> dict[str, Any]:
    candidate_row, shadow_row = _checked_work(candidate), _checked_work(shadow)
    fields = {
        field: {
            "candidate": candidate_row[field],
            "same_state_cold_k4096": shadow_row[field],
            "candidate_fraction_of_shadow": (
                candidate_row[field] / shadow_row[field]
                if shadow_row[field]
                else None
            ),
            "strictly_lower": candidate_row[field] < shadow_row[field],
            "no_greater": candidate_row[field] <= shadow_row[field],
            "equal": candidate_row[field] == shadow_row[field],
            "scientific_gate_field": field in WORK_GATE_FIELDS,
        }
        for field in WORK_FIELDS
    }
    return {
        "fields": fields,
        "gate_fields_strictly_lower": all(
            fields[field]["strictly_lower"] for field in WORK_GATE_FIELDS
        ),
        "gate_fields_no_greater": all(
            fields[field]["no_greater"] for field in WORK_GATE_FIELDS
        ),
    }


def _sum_run_work(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, int], dict[str, int]]:
    candidate, shadow = _new_work(), _new_work()
    for row in rows:
        policies = {item["policy"]: item for item in row["policies"]}
        metrics = policies["candidate"]["metrics"]
        _add_work(candidate, metrics["work"])
        _add_work(shadow, metrics["same_state_shadow_work"])
    return candidate, shadow


def _paired_case_summary(
    runs: Sequence[Mapping[str, Any]], lanes: int, load: float, eta: float
) -> dict[str, Any]:
    members = sorted(
        (row for row in runs if row["lanes"] == lanes and row["load"] == load),
        key=lambda row: row["seed"],
    )
    if [row["seed"] for row in members] != list(ARRIVAL_SEEDS):
        raise RawFixedCapTrajectoryRefusal(
            "case does not contain the registered seed roster"
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
            raise RawFixedCapTrajectoryRefusal("paired mean-flow ratio is undefined")
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
            cm["maximum_queue_recursion_violation"] <= EXECUTION_TOLERANCE
            and rm["maximum_queue_recursion_violation"] <= EXECUTION_TOLERANCE
        )
        flow_identities &= all(
            item["flow_decomposition"]["status"] == "PASS"
            and abs(
                item["flow_decomposition"]["mean_flow_s"]
                - item["metrics"]["mean_flow_s"]
            )
            <= EXECUTION_TOLERANCE
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
    per_case_alpha = FAMILY_ALPHA / PRIMARY_FAMILY_SIZE
    critical = float(student_t.ppf(1.0 - per_case_alpha, df=len(ratios) - 1))
    upper = estimate + critical * standard_error
    flow_pass = upper <= 1.0 + NONINFERIORITY_MARGIN + SCORE_TOLERANCE
    support_pass = bool(
        eta - max_epsilon > SCORE_TOLERANCE
        and eta - max_regret > SCORE_TOLERANCE
    )
    action_pass = bool(action_identity and regret_bounded)
    case_candidate, case_shadow = _sum_run_work(members)
    comparison = _work_comparison(case_candidate, case_shadow)
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
        "candidate_fixed_budget": CANDIDATE_BUDGETS[lanes],
        "cold_reference_budget": REFERENCE_BUDGET,
        "seeds": list(ARRIVAL_SEEDS),
        "seed_count": len(members),
        "paired_seed_rows": seed_rows,
        "paired_mean_flow_ratios": ratios,
        "paired_ratio_mean": estimate,
        "paired_ratio_standard_error": standard_error,
        "one_sided_confidence_level": 1.0 - per_case_alpha,
        "critical_t": critical,
        "paired_ratio_upper_confidence_bound": upper,
        "noninferiority_acceptance_upper": 1.0 + NONINFERIORITY_MARGIN,
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


def _aggregate_work_gates(
    runs: Sequence[Mapping[str, Any]], cases: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    lane_rows = []
    for lanes in LANES:
        members = [row for row in runs if row["lanes"] == lanes]
        candidate, shadow = _sum_run_work(members)
        comparison = _work_comparison(candidate, shadow)
        lane_rows.append(
            {
                "lanes": lanes,
                "candidate": candidate,
                "same_state_cold_k4096_shadow": shadow,
                "comparison": comparison,
                "gate_passed": comparison["gate_fields_strictly_lower"],
            }
        )
    candidate, shadow = _sum_run_work(runs)
    comparison = _work_comparison(candidate, shadow)
    aggregate_pass = comparison["gate_fields_strictly_lower"]
    return {
        "uniform_case_no_regression_gate_passed": all(
            row["case_compute_gate_passed"] for row in cases
        ),
        "by_lane": lane_rows,
        "aggregate": {
            "candidate": candidate,
            "same_state_cold_k4096_shadow": shadow,
            "comparison": comparison,
            "gate_passed": aggregate_pass,
        },
        "gate_passed": bool(
            aggregate_pass
            and all(row["gate_passed"] for row in lane_rows)
            and all(row["case_compute_gate_passed"] for row in cases)
        ),
    }


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
        raise RawFixedCapTrajectoryRefusal(f"trajectory result exists: {output}")
    protocol = _read_object(Path(registration))
    paths = protocol.get("source_paths") or {}
    if set(paths) != {
        "development_registration",
        "development_fixed_sources",
        "development_result",
    }:
        raise RawFixedCapTrajectoryRefusal("registered source paths changed")
    source_contract, _, _ = _validated_development_bundle(paths)
    expected = build_registration(
        development_registration=Path(paths["development_registration"]),
        development_fixed_sources=Path(paths["development_fixed_sources"]),
        development_result=Path(paths["development_result"]),
    )
    _validate_protocol(protocol, expected)
    if source_contract != protocol["development_pass_contract"]:
        raise RawFixedCapTrajectoryRefusal("development PASS contract changed")
    registered_sources = _read_object(Path(fixed_sources))
    if registered_sources != _current_fixed_sources():
        raise RawFixedCapTrajectoryRefusal("registered trajectory sources changed")

    model = old_transition.baseline._model_from_snapshot(
        protocol["physical_complete_frame_model"]
    )
    if _json(model.snapshot()) != protocol["physical_complete_frame_model"]:
        raise RawFixedCapTrajectoryRefusal("frozen model failed exact reconstruction")
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
                raise RawFixedCapTrajectoryRefusal("frozen capacity row changed")
            by_case[key] = row
    if set(by_case) != {(lanes, load) for lanes in LANES for load in LOADS}:
        raise RawFixedCapTrajectoryRefusal("frozen R/load capacity grid changed")
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
        development_registration=Path(paths["development_registration"]),
        development_fixed_sources=Path(paths["development_fixed_sources"]),
        development_result=Path(paths["development_result"]),
    )
    _validate_protocol(protocol, post_expected)
    if (
        _read_object(Path(fixed_sources)) != registered_sources
        or registered_sources != _current_fixed_sources()
    ):
        raise RawFixedCapTrajectoryRefusal(
            "registered trajectory sources changed during execution"
        )
    eta = protocol["lane_minimum_effective_slack_eta"]
    cases = [
        _paired_case_summary(
            runs, lanes, load, float(eta[str(lanes)])
        )
        for lanes in LANES
        for load in LOADS
    ]
    work_gates = _aggregate_work_gates(runs, cases)
    policy_rows = [policy for run in runs for policy in run["policies"]]
    candidate_rows = [row for row in policy_rows if row["policy"] == "candidate"]
    reference_rows = [
        row for row in policy_rows if row["policy"] == "cold_k4096_reference"
    ]
    total_candidate_frames = sum(
        row["metrics"]["frame_count"] for row in candidate_rows
    )
    total_reference_frames = sum(
        row["metrics"]["frame_count"] for row in reference_rows
    )
    checks = {
        "development_pass_protocol_and_fixed_sources_revalidated": True,
        "protocol_and_fixed_sources_revalidated_after_execution": True,
        "fresh_seed_roster_is_disjoint_from_all_prior_arrival_seeds": not (
            set(ARRIVAL_SEEDS) & set(PREVIOUSLY_USED_ARRIVAL_SEEDS)
        ),
        "arrival_tape_count_is_registered_120": len(runs) == len(tasks) == 120,
        "executed_trajectory_count_is_registered_240": len(policy_rows) == 240,
        "one_arrival_generation_per_common_tape": sum(
            run["arrival_tape"]["arrival_generation_count"] for run in runs
        )
        == 120,
        "same_state_shadow_never_executes_a_trajectory": all(
            run["arrival_tape"]["shadow_trajectory_runs"] == 0 for run in runs
        ),
        "every_candidate_frame_has_one_fresh_shadow": sum(
            row["metrics"]["same_state_shadow_calls"] for row in candidate_rows
        )
        == total_candidate_frames,
        "every_policy_fully_drains_and_flow_decomposes": all(
            row["flow_decomposition"]["status"] == "PASS" for row in policy_rows
        ),
        "every_case_execution_integrity_passes": all(
            row["execution_integrity_passed"] for row in cases
        ),
        "no_capacity_recomputation_or_reclassification": True,
    }
    execution_passed = all(checks.values())
    decision = _decision(execution_passed, cases, work_gates["gate_passed"])
    status = "PASS" if decision == PASS_DECISION else (
        "INVALID" if decision == INVALID_DECISION else "FAIL"
    )
    report = {
        "schema_version": 1,
        "gate": GATE,
        "experiment": EXPERIMENT,
        "status": status,
        "scientific_decision": decision,
        "protocol": protocol,
        "checks": checks,
        "case_summaries": cases,
        "runs": runs,
        "work_gates": work_gates,
        "work_accounting": {
            "arrival_tapes": len(runs),
            "arrival_generations": len(runs),
            "executed_trajectory_runs": len(policy_rows),
            "candidate_trajectory_runs": len(candidate_rows),
            "cold_reference_trajectory_runs": len(reference_rows),
            "shadow_trajectory_runs": 0,
            "candidate_selector_calls": total_candidate_frames,
            "same_state_shadow_selector_calls": total_candidate_frames,
            "executed_reference_selector_calls": total_reference_frames,
            "server_pulls": 0,
        },
        "trajectory_transition_passed": decision == PASS_DECISION,
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
    except RawFixedCapTrajectoryRefusal as exc:
        raise SystemExit(f"REFUSED: {exc}") from exc
    print(
        json.dumps(
            {
                "status": payload["status"],
                "experiment": payload["experiment"],
                "scientific_decision": payload.get("scientific_decision"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
