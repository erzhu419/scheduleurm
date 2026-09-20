"""Independent trajectory compute confirmation for cold exact count-space B&B.

Each frame makes one fresh-root call with a K4096 hard cap.  The existing
selector returns as soon as its frontier certificate is exact, so reported LP
and matrix work is the work actually executed rather than the unused cap.

Registration freezes the source chain and protocol without loading the model,
computing capacity, constructing an arrival tape, or calling a selector.
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
import time

from algorithm.count_space_bnb_frame_candidates import select_count_space_bnb_frame
from algorithm.experiments.aligned_frame_flow_decomposition import decompose_flow
from algorithm.experiments.aligned_hardware_count_space_bnb_temporal_incumbent_development import (
    DEFAULT_OUTPUT as DEFAULT_TEMPORAL_RESULT,
    EXPERIMENT as SOURCE_EXPERIMENT,
)
from algorithm.experiments.aligned_hardware_count_space_bnb_trajectory_transition import (
    EXECUTION_TOLERANCE,
    _compact_decomposition,
    _json,
    _selection_certificate_valid,
    _write_new,
)
from algorithm.experiments.aligned_hardware_frame_experiment import poisson_arrivals
from algorithm.experiments.aligned_hardware_frame_model import (
    capacity_certificate,
    load_model,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACTS / (
    "aligned_hardware_count_space_bnb_cold_exact_compute_"
    "confirmation_20260913.json"
)
EXPERIMENT = "aligned_hardware_count_space_bnb_cold_exact_compute_confirmation_v1"
SOURCE_DECISION = "TEMPORAL_INCUMBENT_WORK_FAIL_STOP"
RESTART_SOURCE_DECISION = "EXACTNESS_ESCALATION_WORK_FAIL_STOP"
THIS_SOURCE = (
    "algorithm/experiments/"
    "aligned_hardware_count_space_bnb_cold_exact_compute_confirmation.py"
)

LANES = (8, 16)
LOADS = (0.80, 0.95)
ARRIVAL_SEEDS = tuple(range(9041, 9049))
PREVIOUSLY_USED_ARRIVAL_SEEDS = (
    41,
    42,
    43,
    *range(1041, 1051),
    *range(2041, 2061),
    *range(3041, 3061),
    *range(4041, 4061),
    *range(6041, 6049),
    *range(7041, 7049),
    *range(8041, 8049),
)
ARRIVAL_HORIZON_S = 600.0
EXECUTION_WORKERS = 4
LP_NODE_CAP = 4096
LP_UTILIZATION_LIMIT = 0.05
P95_LP_FRACTION_LIMIT = 0.10
WORK_FIELDS = (
    "lp_calls",
    "lp_matrix_coefficients",
    "candidate_evaluations",
    "residual_fill_option_probes",
    "dual_column_checks",
    "branch_splits",
)

DECISION_RULE = {
    "INVALID_EXECUTION": "one or more registered execution-integrity checks fail",
    "COLD_EXACT_COMPUTE_CONFIRMATION_PASS": (
        "every visited frame closes exactly within one cold K4096 call; actual "
        "LP utilization is below five percent in every R/load case and in "
        "aggregate; and empirical frame-level p95 LP work is at most ten "
        "percent of the cap in every case and in aggregate"
    ),
    "COLD_EXACT_COMPUTE_CONFIRMATION_FAIL_STOP": (
        "execution is valid but one or more frozen exactness or empirical "
        "bounded-compute gates fail"
    ),
}

CLAIM_BOUNDARY = (
    "An independent registered finite-horizon trajectory compute confirmation "
    "on eight fresh Poisson seeds per load for point-model synthetic identical-"
    "lane R8 and R16 systems. It establishes empirical work on these visited "
    "frames only. It is not an all-queue theorem, a performance advantage over "
    "another policy, a stability proof, joint95 physical closure, a physical "
    "multi-GPU measurement, or a deployment decision."
)
PROHIBITED = (
    "post-result changes to seeds, horizon, loads, cap, utilization gates, or "
    "endpoints; restart or warm-start retries; an executed comparison policy; "
    "ordered-action or count-vector enumeration in trajectories; future-arrival "
    "access; paper or README edits"
)


def _paths(output=DEFAULT_OUTPUT):
    output = Path(output)
    return (
        output,
        output.with_name(output.stem + "_protocol.json"),
        output.with_name(output.stem + "_fixed_sources.json"),
    )


def _source_paths(result_path):
    result_path = Path(result_path)
    return (
        result_path,
        result_path.with_name(result_path.stem + "_protocol.json"),
        result_path.with_name(result_path.stem + "_fixed_sources.json"),
    )


def _source_contract(temporal_result=DEFAULT_TEMPORAL_RESULT):
    result_path, protocol_path, fixed_path = _source_paths(temporal_result)
    result = json.loads(result_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    fixed_sources = json.loads(fixed_path.read_text())
    if result.get("protocol") != protocol:
        raise ValueError("temporal-incumbent result does not embed its protocol")
    changed = [
        name
        for name, text in fixed_sources.items()
        if (ROOT / name).read_text() != text
    ]
    if changed:
        raise ValueError(f"frozen temporal-incumbent source changed: {changed}")
    restart_source = protocol.get("source_contract", {})
    expected_cases = {(lanes, load) for lanes in LANES for load in LOADS}
    observed_cases = {
        (row.get("lanes"), row.get("load"))
        for row in result.get("case_summaries", [])
    }
    if (
        result.get("experiment") != SOURCE_EXPERIMENT
        or result.get("status") != "PASS"
        or result.get("scientific_decision") != SOURCE_DECISION
        or result.get("temporal_incumbent_advance") is not False
        or not all(result.get("checks", {}).values())
        or observed_cases != expected_cases
        or restart_source.get("escalation_decision") != RESTART_SOURCE_DECISION
        or restart_source.get("escalation_advance") is not False
    ):
        raise ValueError(
            "cold exact confirmation requires the valid failed restart and "
            "temporal-incumbent development chain"
        )
    return (
        {
            "temporal_result_path": str(result_path.resolve()),
            "temporal_experiment": result["experiment"],
            "temporal_decision": result["scientific_decision"],
            "temporal_advance": result["temporal_incumbent_advance"],
            "restart_experiment": restart_source["escalation_experiment"],
            "restart_decision": restart_source["escalation_decision"],
            "restart_advance": restart_source["escalation_advance"],
            "prior_case_count": len(result["case_summaries"]),
        },
        protocol["model"],
        fixed_sources,
    )


def _current_fixed_sources(upstream_fixed_sources):
    return {
        **upstream_fixed_sources,
        THIS_SOURCE: (ROOT / THIS_SOURCE).read_text(),
    }


def _matrix_envelope_constants(model_snapshot):
    option_count = len(model_snapshot["options"])
    dimension = len(model_snapshot["workloads"])
    coefficients_per_lp = option_count * (1 + dimension + option_count)
    return {
        "option_count": option_count,
        "queue_dimension": dimension,
        "maximum_coefficients_per_lp": coefficients_per_lp,
        "derivation": (
            "option_count * (one lane row + queue_dimension resource rows + "
            "option_count box rows)"
        ),
        "interpretation": (
            "a non-executed worst-case coefficient envelope; actual selector "
            "matrix cells are summed directly and no actual comparison baseline "
            "is claimed"
        ),
    }


def _protocol_constants(model_snapshot, source_contract):
    if set(ARRIVAL_SEEDS) & set(PREVIOUSLY_USED_ARRIVAL_SEEDS):
        raise ValueError("confirmation seeds overlap a consumed trajectory roster")
    matrix = _matrix_envelope_constants(model_snapshot)
    return {
        "experiment": EXPERIMENT,
        "role": "independent cold exact trajectory compute confirmation",
        "source_contract": source_contract,
        "model": _json(model_snapshot),
        "lanes": list(LANES),
        "loads": list(LOADS),
        "arrival_seeds": list(ARRIVAL_SEEDS),
        "previously_used_arrival_seeds": list(PREVIOUSLY_USED_ARRIVAL_SEEDS),
        "seeds_disjoint_from_previous_trajectories": True,
        "arrival_horizon_s": ARRIVAL_HORIZON_S,
        "arrival_model": (
            "independent class-wise Poisson arrivals using stream key "
            "aligned-frame:{seed}:{coordinate}"
        ),
        "fixed_tape_rule": (
            "construct exactly one immutable tape for each (lanes, load, seed) "
            "and use it once for the registered cold trajectory"
        ),
        "selector_rule": {
            "family": "original cold count-space branch-and-bound selector",
            "calls_per_frame": 1,
            "lp_node_cap": LP_NODE_CAP,
            "initialization": "fresh roots and frontier in every frame",
            "early_stop": (
                "the call returns before the cap when the certified frontier "
                "upper bound equals the feasible incumbent score"
            ),
            "restarts": 0,
            "warm_starts": 0,
            "required_exact_in_every_frame": True,
            "accounting": "sum each returned call's actual native work fields",
        },
        "online_information": (
            "each call receives only the current released queue, fixed catalogue, "
            "lane count, and K4096 cap"
        ),
        "dispatch": (
            "global synchronous frames, FIFO within class, with arrivals during "
            "a frame eligible only at the next frame boundary"
        ),
        "stopping_rule": (
            "stop arrivals at 600 seconds and drain the finite tape to completion"
        ),
        "primary_cases": [
            {"lanes": lanes, "load": load, "lp_node_cap": LP_NODE_CAP}
            for lanes in LANES
            for load in LOADS
        ],
        "lp_work_gates": {
            "mean_utilization": {
                "quantity": "sum actual LP calls / (4096 * visited frame count)",
                "strict_upper": LP_UTILIZATION_LIMIT,
                "scope": "every R/load case and aggregate",
            },
            "empirical_p95": {
                "estimator": "nearest-rank ceil(0.95*n), one-indexed",
                "quantity": "actual LP calls in one visited frame / 4096",
                "upper_inclusive": P95_LP_FRACTION_LIMIT,
                "scope": "every R/load case and aggregate",
                "claim": "descriptive empirical tail gate on registered frames",
            },
            "maximum": "reported without a favorable threshold",
        },
        "matrix_work_accounting": {
            **matrix,
            "case_cap_envelope": (
                "visited frame count * 4096 * maximum_coefficients_per_lp"
            ),
            "gate": (
                "actual matrix sum must not exceed the arithmetic cap envelope; "
                "no matrix-utilization threshold is a scientific gate"
            ),
        },
        "registered_gate": [
            "one selector call per visited frame and every call is exact within K4096",
            "all jobs complete; queue recursion, FIFO decomposition, and every "
            "returned certificate are valid",
            "no ordered-action or count-vector enumeration occurs in trajectories",
            "actual LP mean utilization is strictly below five percent in every "
            "case and aggregate",
            "empirical frame-level p95 actual LP work is at most ten percent of "
            "the cap in every case and aggregate",
            "all actual LP and matrix work totals reconstruct from per-frame records",
        ],
        "expected_work": {
            "arrival_tapes": len(LANES) * len(LOADS) * len(ARRIVAL_SEEDS),
            "trajectory_runs": len(LANES) * len(LOADS) * len(ARRIVAL_SEEDS),
            "selector_calls": "exactly one per visited frame",
            "ordered_action_enumerations": 0,
            "count_vector_enumerations": 0,
            "execution_workers": EXECUTION_WORKERS,
        },
        "decision_rule": DECISION_RULE,
        "claim_boundary": CLAIM_BOUNDARY,
        "prohibited": PROHIBITED,
        "timing_is_scientific_gate": False,
    }


def register(output=DEFAULT_OUTPUT, temporal_result=DEFAULT_TEMPORAL_RESULT):
    """Freeze protocol and sources without model, capacity, tape, or selector work."""
    output, protocol_path, fixed_sources_path = _paths(output)
    if any(path.exists() for path in (output, protocol_path, fixed_sources_path)):
        raise ValueError("cold exact compute confirmation artifacts already exist")
    source_contract, source_model, upstream_fixed = _source_contract(temporal_result)
    fixed_sources = _current_fixed_sources(upstream_fixed)
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        **_protocol_constants(source_model, source_contract),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_new(protocol_path, protocol, compact=True)
    _write_new(fixed_sources_path, fixed_sources)
    return protocol


def _nearest_rank(values, quantile):
    values = sorted(values)
    if not values:
        raise ValueError("nearest-rank quantile requires at least one value")
    return values[math.ceil(quantile * len(values)) - 1]


def simulate_cold_exact_frames(
    model,
    arrivals,
    arrival_horizon,
    *,
    audit_limit=4,
):
    """Execute FIFO frames with one fresh-root, exactness-stopped K4096 call."""
    queues = [deque() for _ in model.workloads]
    tape = tuple(sorted(arrivals))
    pending = iter(tape)
    next_arrival = next(pending, None)
    now = 0.0
    flows = []
    frames = []
    audits = []
    selected_counts = Counter()
    totals = {field: 0 for field in WORK_FIELDS}
    lp_calls_by_frame = []
    matrix_cells_by_frame = []
    frame_count = 0
    exact_frames = 0
    valid_frames = 0
    max_recursion_violation = 0
    max_score_upper_violation = 0.0
    max_gap_identity_error = 0.0
    peak_backlog = 0
    backlog_sum = 0
    last_completion = 0.0
    selector_wall_s = 0.0

    while now < arrival_horizon or next_arrival is not None or any(queues):
        while next_arrival is not None and next_arrival[0] <= now:
            arrival_time, coordinate, index = next_arrival
            queues[coordinate].append((arrival_time, index))
            next_arrival = next(pending, None)
        before = tuple(len(queue) for queue in queues)
        started = time.perf_counter()
        selected = select_count_space_bnb_frame(
            model.options,
            before,
            model.lanes,
            LP_NODE_CAP,
        )
        selector_wall_s += time.perf_counter() - started
        certificate_valid = _selection_certificate_valid(
            model, before, model.lanes, LP_NODE_CAP, selected
        )
        frame_count += 1
        exact_frames += int(selected.exact)
        valid_frames += int(certificate_valid)
        lp_calls_by_frame.append(selected.lp_calls)
        matrix_cells_by_frame.append(selected.lp_matrix_coefficients)
        for field in WORK_FIELDS:
            totals[field] += getattr(selected, field)
        max_score_upper_violation = max(
            max_score_upper_violation, selected.score - selected.upper_bound
        )
        max_gap_identity_error = max(
            max_gap_identity_error,
            abs(selected.gap - max(0.0, selected.upper_bound - selected.score)),
        )
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
            raise ValueError("count-space service differs from executed departures")

        arrivals_during = [0] * len(queues)
        while next_arrival is not None and next_arrival[0] <= end:
            arrival_time, coordinate, index = next_arrival
            queues[coordinate].append((arrival_time, index))
            arrivals_during[coordinate] += 1
            next_arrival = next(pending, None)
        after = tuple(len(queue) for queue in queues)
        max_recursion_violation = max(
            max_recursion_violation,
            max(
                abs(q1 - (q0 - departure + arrived))
                for q0, departure, arrived, q1 in zip(
                    before, departures, arrivals_during, after, strict=True
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
            audits.append({
                **frame,
                "arrivals_during_frame": tuple(arrivals_during),
                "queue_after": after,
                "lp_node_cap": LP_NODE_CAP,
                "actual_lp_calls": selected.lp_calls,
                "actual_lp_matrix_coefficients": selected.lp_matrix_coefficients,
                "exact": selected.exact,
                "certificate_valid": certificate_valid,
            })
        now = end

    ordered_flows = sorted(flows)
    return ({
        "lp_node_cap": LP_NODE_CAP,
        "arrival_horizon_s": arrival_horizon,
        "elapsed_including_drain_s": now,
        "job_count": len(tape),
        "completed_jobs": len(flows),
        "makespan_s": last_completion,
        "mean_flow_s": sum(flows) / len(flows) if flows else None,
        "p90_flow_s": (
            ordered_flows[math.ceil(0.9 * len(ordered_flows)) - 1]
            if ordered_flows
            else None
        ),
        "frame_count": frame_count,
        "selector_call_count": frame_count,
        "exact_selection_frame_count": exact_frames,
        "valid_certificate_frame_count": valid_frames,
        "maximum_queue_recursion_violation": max_recursion_violation,
        "maximum_score_upper_violation": max_score_upper_violation,
        "maximum_gap_identity_error": max_gap_identity_error,
        "frame_boundary_peak_backlog": peak_backlog,
        "frame_boundary_mean_backlog": backlog_sum / frame_count,
        "selector_wall_s": selector_wall_s,
        "selector_mean_wall_ms": 1000.0 * selector_wall_s / frame_count,
        "selected_option_counts": dict(selected_counts),
        "actual_work": totals,
        "actual_lp_calls_by_frame": lp_calls_by_frame,
        "actual_lp_matrix_coefficients_by_frame": matrix_cells_by_frame,
        "empirical_p95_actual_lp_calls": _nearest_rank(lp_calls_by_frame, 0.95),
        "maximum_actual_lp_calls": max(lp_calls_by_frame),
        "maximum_actual_lp_matrix_coefficients": max(matrix_cells_by_frame),
        "frame_audit_prefix": audits,
    }, frames, tuple(flows))


def _run_tape(task):
    model, certificate, lanes, load, seed, horizon = task
    lane_model = replace(model, lanes=lanes)
    tape = poisson_arrivals(certificate["arrival_rates"], horizon, seed)
    metrics, frames, flows = simulate_cold_exact_frames(
        lane_model, tape, horizon
    )
    decomposition = decompose_flow(lane_model, tape, frames, horizon)
    arrivals_by_class = [0] * len(model.workloads)
    for _, coordinate, _ in tape:
        arrivals_by_class[coordinate] += 1
    return {
        "lanes": lanes,
        "load": load,
        "seed": seed,
        "arrival_tape": {
            "job_count": len(tape),
            "jobs_by_class": arrivals_by_class,
            "first_arrival_s": tape[0][0] if tape else None,
            "last_arrival_s": tape[-1][0] if tape else None,
            "generation_count": 1,
        },
        "metrics": metrics,
        "flow_decomposition": _compact_decomposition(decomposition),
        "flow_count": len(flows),
    }


def _summarize(runs, lanes, load, matrix_coefficients_per_lp):
    members = sorted(
        (row for row in runs if row["lanes"] == lanes and row["load"] == load),
        key=lambda row: row["seed"],
    )
    frame_count = sum(row["metrics"]["frame_count"] for row in members)
    actual_lp = sum(
        row["metrics"]["actual_work"]["lp_calls"] for row in members
    )
    actual_matrix = sum(
        row["metrics"]["actual_work"]["lp_matrix_coefficients"]
        for row in members
    )
    per_frame_lp = [
        value
        for row in members
        for value in row["metrics"]["actual_lp_calls_by_frame"]
    ]
    per_frame_matrix = [
        value
        for row in members
        for value in row["metrics"]["actual_lp_matrix_coefficients_by_frame"]
    ]
    lp_envelope = LP_NODE_CAP * frame_count
    matrix_envelope = lp_envelope * matrix_coefficients_per_lp
    lp_utilization = actual_lp / lp_envelope
    p95_lp = _nearest_rank(per_frame_lp, 0.95)
    p95_fraction = p95_lp / LP_NODE_CAP
    all_jobs_complete = all(
        row["metrics"]["job_count"]
        == row["metrics"]["completed_jobs"]
        == row["arrival_tape"]["job_count"]
        == row["flow_count"]
        for row in members
    )
    all_recursions = all(
        row["metrics"]["maximum_queue_recursion_violation"] == 0
        for row in members
    )
    all_certificates = all(
        row["metrics"]["valid_certificate_frame_count"]
        == row["metrics"]["frame_count"]
        and row["metrics"]["maximum_score_upper_violation"]
        <= EXECUTION_TOLERANCE
        and row["metrics"]["maximum_gap_identity_error"]
        <= EXECUTION_TOLERANCE
        for row in members
    )
    all_exact = all(
        row["metrics"]["exact_selection_frame_count"]
        == row["metrics"]["frame_count"]
        for row in members
    )
    one_call_per_frame = all(
        row["metrics"]["selector_call_count"] == row["metrics"]["frame_count"]
        for row in members
    )
    per_frame_records_complete = (
        len(per_frame_lp) == len(per_frame_matrix) == frame_count
        and sum(per_frame_lp) == actual_lp
        and sum(per_frame_matrix) == actual_matrix
    )
    all_within_cap = max(per_frame_lp) <= LP_NODE_CAP
    fifo_holds = all(
        row["flow_decomposition"]["status"] == "PASS"
        and abs(
            row["flow_decomposition"]["mean_flow_s"]
            - row["metrics"]["mean_flow_s"]
        ) <= EXECUTION_TOLERANCE
        for row in members
    )
    lp_mean_gate = lp_utilization < LP_UTILIZATION_LIMIT
    lp_p95_gate = p95_fraction <= P95_LP_FRACTION_LIMIT
    matrix_accounting_holds = actual_matrix <= matrix_envelope
    seed_rows = [
        {
            "seed": row["seed"],
            "job_count": row["metrics"]["job_count"],
            "frame_count": row["metrics"]["frame_count"],
            "mean_flow_s": row["metrics"]["mean_flow_s"],
            "actual_lp_calls": row["metrics"]["actual_work"]["lp_calls"],
            "actual_lp_utilization_of_cap": (
                row["metrics"]["actual_work"]["lp_calls"]
                / (LP_NODE_CAP * row["metrics"]["frame_count"])
            ),
            "empirical_p95_actual_lp_calls": row["metrics"][
                "empirical_p95_actual_lp_calls"
            ],
            "maximum_actual_lp_calls": row["metrics"]["maximum_actual_lp_calls"],
            "actual_lp_matrix_coefficients": row["metrics"]["actual_work"][
                "lp_matrix_coefficients"
            ],
        }
        for row in members
    ]
    case_gate = all((
        len(members) == len(ARRIVAL_SEEDS),
        [row["seed"] for row in members] == list(ARRIVAL_SEEDS),
        all_jobs_complete,
        all_recursions,
        all_certificates,
        all_exact,
        one_call_per_frame,
        per_frame_records_complete,
        all_within_cap,
        fifo_holds,
        lp_mean_gate,
        lp_p95_gate,
        matrix_accounting_holds,
    ))
    return {
        "lanes": lanes,
        "load": load,
        "seeds": list(ARRIVAL_SEEDS),
        "seed_count": len(members),
        "seed_rows": seed_rows,
        "frame_count": frame_count,
        "all_jobs_complete": all_jobs_complete,
        "all_queue_recursions_hold": all_recursions,
        "all_selection_certificates_valid": all_certificates,
        "exact_in_every_frame": all_exact,
        "one_cold_selector_call_per_frame": one_call_per_frame,
        "all_fifo_flow_decompositions_hold": fifo_holds,
        "per_frame_work_records_reconstruct_totals": per_frame_records_complete,
        "actual_lp_calls": actual_lp,
        "lp_cap_envelope": lp_envelope,
        "actual_lp_utilization_of_cap": lp_utilization,
        "actual_lp_utilization_strictly_below_five_percent": lp_mean_gate,
        "empirical_p95_actual_lp_calls": p95_lp,
        "empirical_p95_actual_lp_fraction_of_cap": p95_fraction,
        "empirical_p95_at_most_ten_percent_of_cap": lp_p95_gate,
        "maximum_actual_lp_calls": max(per_frame_lp),
        "maximum_actual_lp_fraction_of_cap": max(per_frame_lp) / LP_NODE_CAP,
        "every_frame_within_cap": all_within_cap,
        "actual_lp_matrix_coefficients": actual_matrix,
        "matrix_cap_envelope": matrix_envelope,
        "actual_matrix_fraction_of_nonexecuted_cap_envelope": (
            actual_matrix / matrix_envelope
        ),
        "maximum_actual_lp_matrix_coefficients": max(per_frame_matrix),
        "matrix_accounting_within_cap_envelope": matrix_accounting_holds,
        "case_gate_passed": case_gate,
    }


def _decision(execution_passed, case_summaries, aggregate_gate_passed):
    if not execution_passed:
        return "INVALID_EXECUTION"
    if all(row["case_gate_passed"] for row in case_summaries) and aggregate_gate_passed:
        return "COLD_EXACT_COMPUTE_CONFIRMATION_PASS"
    return "COLD_EXACT_COMPUTE_CONFIRMATION_FAIL_STOP"


def run_registered(output=DEFAULT_OUTPUT):
    """Build fresh fixed tapes, run one cold exact trajectory each, and preserve outcome."""
    output, protocol_path, fixed_sources_path = _paths(output)
    if output.exists():
        raise ValueError("cold exact compute confirmation result already exists")
    protocol = json.loads(protocol_path.read_text())
    source_path = protocol.get("source_contract", {}).get("temporal_result_path")
    if not source_path:
        raise ValueError("registered temporal-incumbent source path is missing")
    source_contract, source_model, upstream_fixed = _source_contract(source_path)
    registered_sources = json.loads(fixed_sources_path.read_text())
    if registered_sources != _current_fixed_sources(upstream_fixed):
        raise ValueError("registered cold exact confirmation sources changed")
    expected = _protocol_constants(source_model, source_contract)
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("registered cold exact confirmation protocol changed")

    model = load_model()
    if _json(model.snapshot()) != source_model:
        raise ValueError("current aligned model differs from the registered source")
    certificates = {}
    capacity_calls = 0
    for lanes in LANES:
        for load in LOADS:
            certificates[(lanes, load)] = capacity_certificate(
                replace(model, lanes=lanes), load
            )
            capacity_calls += 1

    tasks = [
        (
            model,
            certificates[(lanes, load)],
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
    with ProcessPoolExecutor(max_workers=EXECUTION_WORKERS) as pool:
        runs = list(pool.map(_run_tape, tasks))

    matrix_per_lp = protocol["matrix_work_accounting"][
        "maximum_coefficients_per_lp"
    ]
    case_summaries = [
        _summarize(runs, lanes, load, matrix_per_lp)
        for lanes in LANES
        for load in LOADS
    ]
    all_metrics = [row["metrics"] for row in runs]
    aggregate_frames = sum(row["frame_count"] for row in all_metrics)
    aggregate_lp = sum(row["actual_work"]["lp_calls"] for row in all_metrics)
    aggregate_matrix = sum(
        row["actual_work"]["lp_matrix_coefficients"] for row in all_metrics
    )
    all_lp_by_frame = [
        value for row in all_metrics for value in row["actual_lp_calls_by_frame"]
    ]
    all_matrix_by_frame = [
        value
        for row in all_metrics
        for value in row["actual_lp_matrix_coefficients_by_frame"]
    ]
    aggregate_lp_envelope = aggregate_frames * LP_NODE_CAP
    aggregate_matrix_envelope = aggregate_lp_envelope * matrix_per_lp
    aggregate_lp_utilization = aggregate_lp / aggregate_lp_envelope
    aggregate_p95 = _nearest_rank(all_lp_by_frame, 0.95)
    aggregate_lp_gate = aggregate_lp_utilization < LP_UTILIZATION_LIMIT
    aggregate_p95_gate = aggregate_p95 / LP_NODE_CAP <= P95_LP_FRACTION_LIMIT
    aggregate_matrix_accounting = (
        len(all_matrix_by_frame) == aggregate_frames
        and sum(all_matrix_by_frame) == aggregate_matrix
        and aggregate_matrix <= aggregate_matrix_envelope
    )
    aggregate_gate_passed = all((
        aggregate_lp_gate,
        aggregate_p95_gate,
        aggregate_matrix_accounting,
    ))

    expected_runs = len(tasks)
    checks = {
        "capacity_certificates_are_valid": all(
            certificate["capacity_projection_passed"]
            and certificate["conditional_model_drift_certified"]
            for certificate in certificates.values()
        ),
        "fresh_seed_roster_is_disjoint": not (
            set(ARRIVAL_SEEDS) & set(PREVIOUSLY_USED_ARRIVAL_SEEDS)
        ),
        "fixed_tape_roster_is_complete": (
            len(runs) == expected_runs
            and len({(r["lanes"], r["load"], r["seed"]) for r in runs})
            == expected_runs
            and all(r["arrival_tape"]["generation_count"] == 1 for r in runs)
        ),
        "all_registered_jobs_complete": all(
            row["metrics"]["completed_jobs"] == row["metrics"]["job_count"]
            == row["flow_count"]
            for row in runs
        ),
        "queue_recursion_holds": all(
            row["metrics"]["maximum_queue_recursion_violation"] == 0
            for row in runs
        ),
        "every_selection_certificate_is_valid": all(
            row["metrics"]["valid_certificate_frame_count"]
            == row["metrics"]["frame_count"]
            for row in runs
        ),
        "cold_call_is_exact_in_every_frame": all(
            row["metrics"]["exact_selection_frame_count"]
            == row["metrics"]["frame_count"]
            for row in runs
        ),
        "one_selector_call_per_frame": all(
            row["metrics"]["selector_call_count"] == row["metrics"]["frame_count"]
            for row in runs
        ),
        "flow_decomposition_reconstructs_mean_flow": all(
            row["flow_decomposition"]["status"] == "PASS"
            and abs(
                row["flow_decomposition"]["mean_flow_s"]
                - row["metrics"]["mean_flow_s"]
            ) <= EXECUTION_TOLERANCE
            for row in runs
        ),
        "per_frame_work_records_reconstruct_all_totals": (
            len(all_lp_by_frame) == len(all_matrix_by_frame) == aggregate_frames
            and sum(all_lp_by_frame) == aggregate_lp
            and sum(all_matrix_by_frame) == aggregate_matrix
        ),
        "every_frame_stops_within_cap": max(all_lp_by_frame) <= LP_NODE_CAP,
        "registered_run_count_is_exact": len(runs) == expected_runs,
        "capacity_call_count_is_exact": capacity_calls == len(LANES) * len(LOADS),
        "no_ordered_or_count_vector_enumeration": True,
        "registered_sources_unchanged_after_execution": (
            registered_sources == _current_fixed_sources(upstream_fixed)
        ),
    }
    execution_passed = all(checks.values())
    scientific_decision = _decision(
        execution_passed, case_summaries, aggregate_gate_passed
    )
    result = {
        "experiment": EXPERIMENT,
        "status": "PASS" if execution_passed else "FAIL",
        "protocol": protocol,
        "checks": checks,
        "case_summaries": case_summaries,
        "aggregate_gate": {
            "frame_count": aggregate_frames,
            "actual_lp_calls": aggregate_lp,
            "lp_cap_envelope": aggregate_lp_envelope,
            "actual_lp_utilization_of_cap": aggregate_lp_utilization,
            "actual_lp_utilization_strictly_below_five_percent": aggregate_lp_gate,
            "empirical_p95_actual_lp_calls": aggregate_p95,
            "empirical_p95_actual_lp_fraction_of_cap": aggregate_p95 / LP_NODE_CAP,
            "empirical_p95_at_most_ten_percent_of_cap": aggregate_p95_gate,
            "maximum_actual_lp_calls": max(all_lp_by_frame),
            "maximum_actual_lp_fraction_of_cap": max(all_lp_by_frame) / LP_NODE_CAP,
            "actual_lp_matrix_coefficients": aggregate_matrix,
            "matrix_cap_envelope": aggregate_matrix_envelope,
            "actual_matrix_fraction_of_nonexecuted_cap_envelope": (
                aggregate_matrix / aggregate_matrix_envelope
            ),
            "maximum_actual_lp_matrix_coefficients": max(all_matrix_by_frame),
            "matrix_accounting_within_cap_envelope": aggregate_matrix_accounting,
            "aggregate_gate_passed": aggregate_gate_passed,
        },
        "work_accounting": {
            "arrival_tapes": len(runs),
            "trajectory_runs": len(runs),
            "selector_calls": aggregate_frames,
            "actual_lp_calls": aggregate_lp,
            "actual_lp_matrix_coefficients": aggregate_matrix,
            "candidate_evaluations": sum(
                row["actual_work"]["candidate_evaluations"] for row in all_metrics
            ),
            "residual_fill_option_probes": sum(
                row["actual_work"]["residual_fill_option_probes"]
                for row in all_metrics
            ),
            "dual_column_checks": sum(
                row["actual_work"]["dual_column_checks"] for row in all_metrics
            ),
            "branch_splits": sum(
                row["actual_work"]["branch_splits"] for row in all_metrics
            ),
            "restart_calls": 0,
            "warm_start_calls": 0,
            "executed_comparison_policy_runs": 0,
            "ordered_action_enumerations": 0,
            "count_vector_enumerations": 0,
            "arrival_tape_generation_calls": len(runs),
        },
        "runs": runs,
        "scientific_decision": scientific_decision,
        "cold_exact_compute_confirmed": (
            scientific_decision == "COLD_EXACT_COMPUTE_CONFIRMATION_PASS"
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "limitations": [
            "The LP utilization and p95 statements are empirical summaries of the registered visited frames, not worst-case complexity bounds.",
            "The matrix denominator is a non-executed cap envelope, not measured work from a competing baseline.",
            "The finite 600-second point-model tapes are drained completely but do not establish steady-state or physical performance.",
        ],
        "next_step": (
            "Preserve the registered result for code-first synthesis while the "
            "separate joint95 physical collection remains open."
        ),
        "timing": {
            "workers": EXECUTION_WORKERS,
            "wall_s": time.perf_counter() - started,
            "timing_is_scientific_gate": False,
        },
    }
    _write_new(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--register", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--temporal-result", type=Path, default=DEFAULT_TEMPORAL_RESULT
    )
    args = parser.parse_args(argv)
    payload = (
        register(args.output, args.temporal_result)
        if args.register
        else run_registered(args.output)
    )
    print(json.dumps({
        "experiment": payload["experiment"],
        "status": payload.get("status", "REGISTERED"),
        "scientific_decision": payload.get("scientific_decision"),
        "output": str(args.output),
    }))


if __name__ == "__main__":
    main()
