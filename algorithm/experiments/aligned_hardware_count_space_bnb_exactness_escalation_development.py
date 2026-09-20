"""Registered development test of exactness-triggered B&B restarts.

R8 starts at K36 and R16 at K72.  A non-exact selection causes a fresh
search from the root at the next doubled budget, up to K4096.  Every attempt's
actual LP and matrix work is charged.  The adaptive trajectory is compared on
the same fresh Poisson tape with one same-family K4096 exact-reference run.

Registration only validates and freezes sources and protocol.  It does not
load the model, compute capacity, construct an arrival tape, or call a selector.
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

from algorithm.count_space_bnb_frame_candidates import (
    select_count_space_bnb_frame,
)
from algorithm.experiments.aligned_frame_flow_decomposition import decompose_flow
from algorithm.experiments.aligned_hardware_count_space_bnb_trajectory_transition import (
    DEFAULT_OUTPUT as DEFAULT_TRANSITION_RESULT,
    EXECUTION_TOLERANCE,
    EXPERIMENT as SOURCE_EXPERIMENT,
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
    "aligned_hardware_count_space_bnb_exactness_escalation_"
    "development_20260913.json"
)
EXPERIMENT = "aligned_hardware_count_space_bnb_exactness_escalation_development_v1"
SOURCE_DECISION = "COUNT_SPACE_TRAJECTORY_TRANSITION_FAIL_STOP"
THIS_SOURCE = (
    "algorithm/experiments/"
    "aligned_hardware_count_space_bnb_exactness_escalation_development.py"
)

LANE_START_BUDGETS = {8: 36, 16: 72}
REFERENCE_LP_BUDGET = 4096
LOADS = (0.80, 0.95)
ARRIVAL_SEEDS = tuple(range(7041, 7049))
PREVIOUSLY_USED_ARRIVAL_SEEDS = (
    41,
    42,
    43,
    *range(1041, 1051),
    *range(2041, 2061),
    *range(3041, 3061),
    *range(4041, 4061),
    *range(6041, 6049),
)
ARRIVAL_HORIZON_S = 600.0
EXECUTION_WORKERS = 4
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
    "EXACTNESS_ESCALATION_PASS": (
        "every adaptive frame is exact and identical to the K4096 reference, "
        "and charged adaptive LP calls and matrix cells are strictly smaller "
        "in every R/load case and in aggregate"
    ),
    "EXACTNESS_ESCALATION_WORK_FAIL_STOP": (
        "execution is valid but one or more registered exactness, identity, or "
        "strict-work gates fail"
    ),
}

CLAIM_BOUNDARY = (
    "A registered finite-horizon development comparison on eight fresh Poisson "
    "seeds per load for synthetic identical-lane R8 and R16 models.  It tests "
    "fresh-root budget escalation on these tapes only; it is not an all-queue "
    "result, stability proof, physical multi-GPU measurement, or deployment decision."
)
PROHIBITED = (
    "post-result changes to seeds, horizon, loads, budget schedules, reference, "
    "or endpoints; treating restarts as cumulative search; ordered-action or "
    "count-vector enumeration; future-arrival access; paper or README edits"
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


def _source_contract(transition_result=DEFAULT_TRANSITION_RESULT):
    result_path, protocol_path, fixed_path = _source_paths(transition_result)
    result = json.loads(result_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    fixed_sources = json.loads(fixed_path.read_text())
    if result.get("protocol") != protocol:
        raise ValueError("trajectory transition does not embed its protocol")
    changed = [
        name
        for name, text in fixed_sources.items()
        if (ROOT / name).read_text() != text
    ]
    if changed:
        raise ValueError(f"frozen trajectory transition source changed: {changed}")
    if (
        result.get("experiment") != SOURCE_EXPERIMENT
        or result.get("status") != "PASS"
        or result.get("scientific_decision") != SOURCE_DECISION
        or result.get("trajectory_transition_advance") is not False
        or not all(result.get("checks", {}).values())
    ):
        raise ValueError("exactness escalation requires the valid failed transition")
    prior_cases = []
    for row in result.get("case_summaries", []):
        prior_cases.append({
            "lanes": row["lanes"],
            "load": row["load"],
            "fixed_lp_budget": row["fixed_lp_budget"],
            "paired_mean_flow_noninferiority_passed": row[
                "paired_mean_flow_noninferiority_passed"
            ],
            "trajectory_support_gate_passed": row[
                "trajectory_support_gate_passed"
            ],
            "case_gate_passed": row["case_gate_passed"],
        })
    expected_pairs = {
        (lanes, load, budget)
        for lanes, budget in LANE_START_BUDGETS.items()
        for load in LOADS
    }
    observed_pairs = {
        (row["lanes"], row["load"], row["fixed_lp_budget"])
        for row in prior_cases
    }
    if observed_pairs != expected_pairs:
        raise ValueError("failed transition lane, load, or starting budgets changed")
    return (
        {
            "trajectory_result_path": str(result_path.resolve()),
            "trajectory_experiment": result["experiment"],
            "trajectory_decision": result["scientific_decision"],
            "trajectory_advance": result["trajectory_transition_advance"],
            "prior_case_gates": prior_cases,
        },
        protocol["model"],
        fixed_sources,
    )


def _current_fixed_sources(upstream_fixed_sources):
    return {
        **upstream_fixed_sources,
        THIS_SOURCE: (ROOT / THIS_SOURCE).read_text(),
    }


def _budget_schedule(start_budget):
    if int(start_budget) != start_budget or not 0 < start_budget <= REFERENCE_LP_BUDGET:
        raise ValueError("starting budget must be a positive integer at most K4096")
    schedule = [int(start_budget)]
    while schedule[-1] < REFERENCE_LP_BUDGET:
        schedule.append(min(2 * schedule[-1], REFERENCE_LP_BUDGET))
    return tuple(schedule)


def _protocol_constants(model_snapshot, source_contract):
    if set(ARRIVAL_SEEDS) & set(PREVIOUSLY_USED_ARRIVAL_SEEDS):
        raise ValueError("escalation seeds overlap a previously used trajectory roster")
    return {
        "experiment": EXPERIMENT,
        "role": "registered development exactness-triggered budget escalation",
        "source_contract": source_contract,
        "model": _json(model_snapshot),
        "lane_start_budgets": [
            {
                "lanes": lanes,
                "start_lp_budget": budget,
                "restart_budget_schedule": list(_budget_schedule(budget)),
            }
            for lanes, budget in LANE_START_BUDGETS.items()
        ],
        "escalation_rule": {
            "trigger": "restart only when the returned selection has exact=false",
            "restart": (
                "each attempt starts a new count-space B&B search from all roots; "
                "no frontier or work is reused"
            ),
            "next_budget": "min(2 * current_budget, 4096)",
            "stop": "execute the first exact selection; K4096 is the hard cap",
            "accounting": (
                "sum actual work reported by every attempted search, including "
                "all failed lower-budget restarts"
            ),
        },
        "reference": {
            "family": "the same count-space branch-and-bound selector",
            "lp_node_budget": REFERENCE_LP_BUDGET,
            "required_exact_in_every_frame": True,
            "accounting": "actual reported work from one K4096 call per frame",
        },
        "loads": list(LOADS),
        "arrival_seeds": list(ARRIVAL_SEEDS),
        "previously_used_arrival_seeds": list(PREVIOUSLY_USED_ARRIVAL_SEEDS),
        "seeds_disjoint_from_previous_trajectories": True,
        "arrival_horizon_s": ARRIVAL_HORIZON_S,
        "arrival_model": (
            "independent class-wise Poisson arrivals using stream key "
            "aligned-frame:{seed}:{coordinate}"
        ),
        "common_tape_rule": (
            "construct one tape for each (lanes, load, seed) and pass its exact "
            "contents to adaptive and K4096 reference runs"
        ),
        "online_information": (
            "each selector attempt receives only current released queues, lanes, "
            "fixed catalogue, and the attempt budget"
        ),
        "dispatch": (
            "global synchronous frames, FIFO within class, with arrivals during "
            "a frame eligible at the next frame boundary"
        ),
        "stopping_rule": (
            "stop arrivals at 600 seconds and drain the common finite tape to completion"
        ),
        "primary_cases": [
            {"lanes": lanes, "load": load, "start_lp_budget": budget}
            for lanes, budget in LANE_START_BUDGETS.items()
            for load in LOADS
        ],
        "registered_gate": [
            "the adaptive final selection is exact in every visited frame",
            "adaptive and K4096 reference frame actions are exactly identical",
            "adaptive and reference per-job flow sequences are exactly identical",
            "charged adaptive LP calls are strictly below reference in every case",
            "charged adaptive matrix cells are strictly below reference in every case",
            "both charged work totals are also strictly below reference in aggregate",
            "all jobs complete and queue recursion, FIFO decomposition, and every "
            "attempt certificate are valid",
        ],
        "expected_work": {
            "arrival_tapes": len(LANE_START_BUDGETS) * len(LOADS) * len(ARRIVAL_SEEDS),
            "trajectory_runs": (
                len(LANE_START_BUDGETS) * len(LOADS) * len(ARRIVAL_SEEDS) * 2
            ),
            "selector_attempts": "reported from actual adaptive restarts and frames",
            "ordered_action_enumerations": 0,
            "count_vector_enumerations": 0,
            "execution_workers": EXECUTION_WORKERS,
        },
        "decision_rule": DECISION_RULE,
        "claim_boundary": CLAIM_BOUNDARY,
        "prohibited": PROHIBITED,
        "timing_is_scientific_gate": False,
    }


def register(output=DEFAULT_OUTPUT, transition_result=DEFAULT_TRANSITION_RESULT):
    """Freeze protocol and sources without model, capacity, tape, or selector work."""
    output, protocol_path, fixed_sources_path = _paths(output)
    if any(path.exists() for path in (output, protocol_path, fixed_sources_path)):
        raise ValueError("exactness-escalation artifacts already exist")
    source_contract, source_model, upstream_fixed = _source_contract(transition_result)
    fixed_sources = _current_fixed_sources(upstream_fixed)
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        **_protocol_constants(source_model, source_contract),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_new(protocol_path, protocol, compact=True)
    _write_new(fixed_sources_path, fixed_sources)
    return protocol


def _select_exact_with_restarts(model, queue, start_budget):
    attempts = []
    charged = {field: 0 for field in WORK_FIELDS}
    selected = None
    for budget in _budget_schedule(start_budget):
        started = time.perf_counter()
        selected = select_count_space_bnb_frame(
            model.options,
            queue,
            model.lanes,
            budget,
        )
        wall_s = time.perf_counter() - started
        valid = _selection_certificate_valid(
            model, queue, model.lanes, budget, selected
        )
        for field in WORK_FIELDS:
            charged[field] += getattr(selected, field)
        attempts.append({
            "lp_node_budget": budget,
            "lp_calls": selected.lp_calls,
            "lp_matrix_coefficients": selected.lp_matrix_coefficients,
            "candidate_evaluations": selected.candidate_evaluations,
            "exact": selected.exact,
            "certificate_valid": valid,
            "selector_wall_s": wall_s,
        })
        if selected.exact:
            break
    if selected is None:
        raise ArithmeticError("empty escalation schedule")
    return selected, attempts, charged


def simulate_exactness_escalation_frames(
    model,
    arrivals,
    arrival_horizon,
    start_budget,
    *,
    audit_limit=4,
):
    """Run synchronous FIFO frames with fresh-root exactness escalation."""
    queues = [deque() for _ in model.workloads]
    tape = tuple(sorted(arrivals))
    pending = iter(tape)
    next_arrival = next(pending, None)
    now = 0.0
    frame_count = 0
    flows = []
    last_completion = 0.0
    peak_backlog = 0
    backlog_sum = 0
    exact_frames = 0
    valid_final_frames = 0
    valid_attempts = 0
    selector_attempts = 0
    escalated_frames = 0
    max_recursion_violation = 0
    max_score_upper_violation = 0.0
    max_gap_identity_error = 0.0
    max_final_budget = 0
    selected_counts = Counter()
    final_budget_counts = Counter()
    attempt_budget_counts = Counter()
    totals = {field: 0 for field in WORK_FIELDS}
    frames = []
    audits = []
    selector_wall_s = 0.0

    while now < arrival_horizon or next_arrival is not None or any(queues):
        while next_arrival is not None and next_arrival[0] <= now:
            arrival_time, coordinate, index = next_arrival
            queues[coordinate].append((arrival_time, index))
            next_arrival = next(pending, None)
        before = tuple(len(queue) for queue in queues)
        selected, attempts, charged = _select_exact_with_restarts(
            model, before, start_budget
        )
        frame_count += 1
        selector_attempts += len(attempts)
        escalated_frames += int(len(attempts) > 1)
        exact_frames += int(selected.exact)
        valid_final_frames += int(attempts[-1]["certificate_valid"])
        valid_attempts += sum(row["certificate_valid"] for row in attempts)
        selector_wall_s += sum(row["selector_wall_s"] for row in attempts)
        final_budget = attempts[-1]["lp_node_budget"]
        max_final_budget = max(max_final_budget, final_budget)
        final_budget_counts[str(final_budget)] += 1
        for row in attempts:
            attempt_budget_counts[str(row["lp_node_budget"])] += 1
        for field in WORK_FIELDS:
            totals[field] += charged[field]

        max_score_upper_violation = max(
            max_score_upper_violation, selected.score - selected.upper_bound
        )
        max_gap_identity_error = max(
            max_gap_identity_error,
            abs(selected.gap - max(0.0, selected.upper_bound - selected.score)),
        )
        peak_backlog = max(peak_backlog, sum(before))
        backlog_sum += sum(before)

        duration = selected.duration
        end = now + duration
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

        during = [0] * len(queues)
        while next_arrival is not None and next_arrival[0] <= end:
            arrival_time, coordinate, index = next_arrival
            queues[coordinate].append((arrival_time, index))
            during[coordinate] += 1
            next_arrival = next(pending, None)
        after = tuple(len(queue) for queue in queues)
        max_recursion_violation = max(
            max_recursion_violation,
            max(
                abs(q1 - (q0 - departure + arrivals_during))
                for q0, departure, arrivals_during, q1 in zip(
                    before, departures, during, after, strict=True
                )
            ),
        )
        frame = {
            "start_s": now,
            "duration_s": duration,
            "queue_before": before,
            "departures": tuple(departures),
            "option_indices": selected.option_indices,
        }
        frames.append(frame)
        if len(audits) < audit_limit:
            audits.append({
                **frame,
                "arrivals_during_frame": tuple(during),
                "queue_after": after,
                "final_budget": final_budget,
                "attempts": attempts,
                "exact": selected.exact,
            })
        now = end

    ordered_flows = sorted(flows)
    metrics = {
        "start_lp_budget": start_budget,
        "hard_cap_lp_budget": REFERENCE_LP_BUDGET,
        "restart_budget_schedule": list(_budget_schedule(start_budget)),
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
        "exact_selection_frame_count": exact_frames,
        "valid_final_certificate_frame_count": valid_final_frames,
        "selector_attempt_count": selector_attempts,
        "valid_certificate_attempt_count": valid_attempts,
        "restart_count": selector_attempts - frame_count,
        "escalated_frame_count": escalated_frames,
        "maximum_final_lp_budget": max_final_budget,
        "final_budget_frame_counts": dict(final_budget_counts),
        "attempt_budget_counts": dict(attempt_budget_counts),
        "maximum_queue_recursion_violation": max_recursion_violation,
        "maximum_score_upper_violation": max_score_upper_violation,
        "maximum_gap_identity_error": max_gap_identity_error,
        "frame_boundary_peak_backlog": peak_backlog,
        "frame_boundary_mean_backlog": backlog_sum / frame_count,
        "selector_wall_s": selector_wall_s,
        "selector_mean_wall_ms_per_attempt": (
            1000.0 * selector_wall_s / selector_attempts
        ),
        "selected_option_counts": dict(selected_counts),
        "charged_work": totals,
        "frame_audit_prefix": audits,
    }
    return metrics, frames, tuple(flows)


def _run_tape(task):
    model, certificate, lanes, load, seed, horizon, start_budget = task
    lane_model = replace(model, lanes=lanes)
    tape = poisson_arrivals(certificate["arrival_rates"], horizon, seed)
    adaptive_metrics, adaptive_frames, adaptive_flows = (
        simulate_exactness_escalation_frames(
            lane_model, tape, horizon, start_budget
        )
    )
    reference_metrics, reference_frames, reference_flows = (
        simulate_exactness_escalation_frames(
            lane_model, tape, horizon, REFERENCE_LP_BUDGET
        )
    )
    adaptive_decomposition = decompose_flow(
        lane_model, tape, adaptive_frames, horizon
    )
    reference_decomposition = decompose_flow(
        lane_model, tape, reference_frames, horizon
    )
    policies = [
        {
            "policy": "exactness_escalation",
            "metrics": adaptive_metrics,
            "flow_decomposition": _compact_decomposition(adaptive_decomposition),
        },
        {
            "policy": "exact_reference",
            "metrics": reference_metrics,
            "flow_decomposition": _compact_decomposition(reference_decomposition),
        },
    ]
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
            "shared_policy_count": len(policies),
        },
        "deterministic_action_trace_exactly_equal": (
            adaptive_frames == reference_frames
        ),
        "per_job_flow_sequence_exactly_equal": adaptive_flows == reference_flows,
        "policies": policies,
    }


def _case_summary(runs, lanes, load):
    members = sorted(
        (row for row in runs if row["lanes"] == lanes and row["load"] == load),
        key=lambda row: row["seed"],
    )
    seed_rows = []
    adaptive_lp_calls = 0
    reference_lp_calls = 0
    adaptive_matrix_cells = 0
    reference_matrix_cells = 0
    all_jobs_complete = True
    all_recursions = True
    all_certificates = True
    adaptive_exact = True
    reference_exact = True
    action_identity = True
    flow_identity = True
    all_fifo_decompositions = True
    for row in members:
        policies = {item["policy"]: item for item in row["policies"]}
        adaptive = policies["exactness_escalation"]
        reference = policies["exact_reference"]
        am = adaptive["metrics"]
        rm = reference["metrics"]
        aw = am["charged_work"]
        rw = rm["charged_work"]
        adaptive_lp_calls += aw["lp_calls"]
        reference_lp_calls += rw["lp_calls"]
        adaptive_matrix_cells += aw["lp_matrix_coefficients"]
        reference_matrix_cells += rw["lp_matrix_coefficients"]
        all_jobs_complete &= (
            am["job_count"] == am["completed_jobs"] == rm["job_count"]
            and rm["job_count"] == rm["completed_jobs"]
            and am["job_count"] == row["arrival_tape"]["job_count"]
        )
        all_recursions &= (
            am["maximum_queue_recursion_violation"] == 0
            and rm["maximum_queue_recursion_violation"] == 0
        )
        all_certificates &= (
            am["valid_certificate_attempt_count"] == am["selector_attempt_count"]
            and rm["valid_certificate_attempt_count"] == rm["selector_attempt_count"]
            and am["valid_final_certificate_frame_count"] == am["frame_count"]
            and rm["valid_final_certificate_frame_count"] == rm["frame_count"]
        )
        adaptive_exact &= am["exact_selection_frame_count"] == am["frame_count"]
        reference_exact &= rm["exact_selection_frame_count"] == rm["frame_count"]
        action_identity &= row["deterministic_action_trace_exactly_equal"]
        flow_identity &= (
            row["per_job_flow_sequence_exactly_equal"]
            and am["mean_flow_s"] == rm["mean_flow_s"]
            and am["p90_flow_s"] == rm["p90_flow_s"]
            and am["makespan_s"] == rm["makespan_s"]
        )
        all_fifo_decompositions &= all(
            item["flow_decomposition"]["status"] == "PASS"
            and abs(
                item["flow_decomposition"]["mean_flow_s"]
                - item["metrics"]["mean_flow_s"]
            ) <= EXECUTION_TOLERANCE
            for item in policies.values()
        )
        seed_rows.append({
            "seed": row["seed"],
            "job_count": am["job_count"],
            "mean_flow_s": am["mean_flow_s"],
            "frame_count": am["frame_count"],
            "adaptive_selector_attempts": am["selector_attempt_count"],
            "adaptive_restart_count": am["restart_count"],
            "adaptive_escalated_frame_count": am["escalated_frame_count"],
            "adaptive_maximum_final_lp_budget": am["maximum_final_lp_budget"],
            "adaptive_lp_calls": aw["lp_calls"],
            "reference_lp_calls": rw["lp_calls"],
            "adaptive_lp_matrix_coefficients": aw["lp_matrix_coefficients"],
            "reference_lp_matrix_coefficients": rw["lp_matrix_coefficients"],
            "deterministic_action_trace_exactly_equal": row[
                "deterministic_action_trace_exactly_equal"
            ],
            "per_job_flow_sequence_exactly_equal": row[
                "per_job_flow_sequence_exactly_equal"
            ],
        })
    lp_saved = reference_lp_calls - adaptive_lp_calls
    matrix_saved = reference_matrix_cells - adaptive_matrix_cells
    lp_gate = adaptive_lp_calls < reference_lp_calls
    matrix_gate = adaptive_matrix_cells < reference_matrix_cells
    return {
        "lanes": lanes,
        "load": load,
        "start_lp_budget": LANE_START_BUDGETS[lanes],
        "restart_budget_schedule": list(_budget_schedule(LANE_START_BUDGETS[lanes])),
        "reference_lp_budget": REFERENCE_LP_BUDGET,
        "seeds": list(ARRIVAL_SEEDS),
        "seed_count": len(members),
        "seed_rows": seed_rows,
        "adaptive_final_exact_in_every_frame": adaptive_exact,
        "reference_exact_in_every_frame": reference_exact,
        "deterministic_action_traces_exactly_equal": action_identity,
        "per_job_flow_sequences_exactly_equal": flow_identity,
        "all_jobs_complete": all_jobs_complete,
        "all_queue_recursions_hold": all_recursions,
        "all_attempt_certificates_valid": all_certificates,
        "all_fifo_flow_decompositions_hold": all_fifo_decompositions,
        "adaptive_lp_calls": adaptive_lp_calls,
        "reference_lp_calls": reference_lp_calls,
        "lp_calls_saved": lp_saved,
        "adaptive_lp_calls_strictly_below_reference": lp_gate,
        "adaptive_lp_matrix_coefficients": adaptive_matrix_cells,
        "reference_lp_matrix_coefficients": reference_matrix_cells,
        "lp_matrix_coefficients_saved": matrix_saved,
        "adaptive_matrix_cells_strictly_below_reference": matrix_gate,
        "case_gate_passed": all((
            adaptive_exact,
            reference_exact,
            action_identity,
            flow_identity,
            all_jobs_complete,
            all_recursions,
            all_certificates,
            all_fifo_decompositions,
            lp_gate,
            matrix_gate,
        )),
    }


def _decision(execution_passed, case_summaries, aggregate_work_gate):
    if not execution_passed:
        return "INVALID_EXECUTION"
    if all(row["case_gate_passed"] for row in case_summaries) and aggregate_work_gate:
        return "EXACTNESS_ESCALATION_PASS"
    return "EXACTNESS_ESCALATION_WORK_FAIL_STOP"


def run_registered(output=DEFAULT_OUTPUT):
    """Build fresh common tapes, run adaptive/reference pairs, and preserve outcome."""
    output, protocol_path, fixed_sources_path = _paths(output)
    if output.exists():
        raise ValueError("exactness-escalation result already exists")
    protocol = json.loads(protocol_path.read_text())
    source_path = protocol.get("source_contract", {}).get("trajectory_result_path")
    if not source_path:
        raise ValueError("registered transition source path is missing")
    source_contract, source_model, upstream_fixed = _source_contract(source_path)
    registered_sources = json.loads(fixed_sources_path.read_text())
    if registered_sources != _current_fixed_sources(upstream_fixed):
        raise ValueError("registered exactness-escalation sources changed")
    expected = _protocol_constants(source_model, source_contract)
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("registered exactness-escalation protocol changed")

    model = load_model()
    if _json(model.snapshot()) != source_model:
        raise ValueError("current aligned model differs from the registered source")
    certificates = {}
    capacity_calls = 0
    for lanes in LANE_START_BUDGETS:
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
            start_budget,
        )
        for lanes, start_budget in LANE_START_BUDGETS.items()
        for load in LOADS
        for seed in ARRIVAL_SEEDS
    ]
    started = time.perf_counter()
    runs = []
    with ProcessPoolExecutor(max_workers=EXECUTION_WORKERS) as pool:
        runs.extend(pool.map(_run_tape, tasks))

    case_summaries = [
        _case_summary(runs, lanes, load)
        for lanes in LANE_START_BUDGETS
        for load in LOADS
    ]
    policy_runs = [policy for row in runs for policy in row["policies"]]
    adaptive_runs = [
        row for row in policy_runs if row["policy"] == "exactness_escalation"
    ]
    reference_runs = [row for row in policy_runs if row["policy"] == "exact_reference"]
    aggregate_adaptive_lp = sum(
        row["metrics"]["charged_work"]["lp_calls"] for row in adaptive_runs
    )
    aggregate_reference_lp = sum(
        row["metrics"]["charged_work"]["lp_calls"] for row in reference_runs
    )
    aggregate_adaptive_matrix = sum(
        row["metrics"]["charged_work"]["lp_matrix_coefficients"]
        for row in adaptive_runs
    )
    aggregate_reference_matrix = sum(
        row["metrics"]["charged_work"]["lp_matrix_coefficients"]
        for row in reference_runs
    )
    aggregate_lp_gate = aggregate_adaptive_lp < aggregate_reference_lp
    aggregate_matrix_gate = aggregate_adaptive_matrix < aggregate_reference_matrix
    aggregate_work_gate = aggregate_lp_gate and aggregate_matrix_gate

    expected_tapes = len(tasks)
    expected_policy_runs = 2 * expected_tapes
    checks = {
        "fresh_seed_roster_is_disjoint": not (
            set(ARRIVAL_SEEDS) & set(PREVIOUSLY_USED_ARRIVAL_SEEDS)
        ),
        "common_tape_roster_is_complete": len(runs) == expected_tapes and all(
            row["arrival_tape"]["shared_policy_count"] == 2 for row in runs
        ),
        "all_registered_jobs_complete": all(
            row["metrics"]["job_count"] == row["metrics"]["completed_jobs"]
            for row in policy_runs
        ),
        "queue_recursion_holds": all(
            row["metrics"]["maximum_queue_recursion_violation"] == 0
            for row in policy_runs
        ),
        "every_attempt_certificate_is_valid": all(
            row["metrics"]["valid_certificate_attempt_count"]
            == row["metrics"]["selector_attempt_count"]
            for row in policy_runs
        ),
        "flow_decomposition_reconstructs_mean_flow": all(
            row["flow_decomposition"]["status"] == "PASS"
            and abs(
                row["flow_decomposition"]["mean_flow_s"]
                - row["metrics"]["mean_flow_s"]
            ) <= EXECUTION_TOLERANCE
            for row in policy_runs
        ),
        "reference_is_exact_in_every_frame": all(
            row["metrics"]["exact_selection_frame_count"]
            == row["metrics"]["frame_count"]
            for row in reference_runs
        ),
        "registered_run_count_is_exact": len(policy_runs) == expected_policy_runs,
        "capacity_call_count_is_exact": capacity_calls == len(LANE_START_BUDGETS) * len(LOADS),
        "no_ordered_or_count_vector_enumeration": True,
        "registered_sources_unchanged_after_execution": (
            registered_sources == _current_fixed_sources(upstream_fixed)
        ),
    }
    execution_passed = all(checks.values())
    scientific_decision = _decision(
        execution_passed, case_summaries, aggregate_work_gate
    )
    elapsed = time.perf_counter() - started
    result = {
        "experiment": EXPERIMENT,
        "status": "PASS" if execution_passed else "FAIL",
        "protocol": protocol,
        "checks": checks,
        "case_summaries": case_summaries,
        "aggregate_gate": {
            "adaptive_lp_calls": aggregate_adaptive_lp,
            "reference_lp_calls": aggregate_reference_lp,
            "lp_calls_saved": aggregate_reference_lp - aggregate_adaptive_lp,
            "adaptive_lp_calls_strictly_below_reference": aggregate_lp_gate,
            "adaptive_lp_matrix_coefficients": aggregate_adaptive_matrix,
            "reference_lp_matrix_coefficients": aggregate_reference_matrix,
            "lp_matrix_coefficients_saved": (
                aggregate_reference_matrix - aggregate_adaptive_matrix
            ),
            "adaptive_matrix_cells_strictly_below_reference": aggregate_matrix_gate,
            "aggregate_work_gate_passed": aggregate_work_gate,
        },
        "work_accounting": {
            "arrival_tapes": len(runs),
            "trajectory_runs": len(policy_runs),
            "adaptive_selector_attempts": sum(
                row["metrics"]["selector_attempt_count"] for row in adaptive_runs
            ),
            "reference_selector_attempts": sum(
                row["metrics"]["selector_attempt_count"] for row in reference_runs
            ),
            "adaptive_restarts": sum(
                row["metrics"]["restart_count"] for row in adaptive_runs
            ),
            "adaptive_escalated_frames": sum(
                row["metrics"]["escalated_frame_count"] for row in adaptive_runs
            ),
            "adaptive_lp_calls": aggregate_adaptive_lp,
            "reference_lp_calls": aggregate_reference_lp,
            "adaptive_lp_matrix_coefficients": aggregate_adaptive_matrix,
            "reference_lp_matrix_coefficients": aggregate_reference_matrix,
            "ordered_action_enumerations": 0,
            "count_vector_enumerations": 0,
            "arrival_tape_generation_calls": len(runs),
        },
        "runs": runs,
        "scientific_decision": scientific_decision,
        "exactness_escalation_advance": (
            scientific_decision == "EXACTNESS_ESCALATION_PASS"
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "limitations": [
            "The finite seed roster cannot establish an all-queue guarantee.",
            "Wall-clock selector timing is diagnostic and is not a scientific gate.",
            "Fresh-root retries intentionally charge repeated search work.",
        ],
        "next_step": (
            "Preserve this registered outcome without changing seeds, schedules, "
            "reference accounting, or gates."
        ),
        "timing": {
            "workers": EXECUTION_WORKERS,
            "wall_s": elapsed,
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
        "--transition-result", type=Path, default=DEFAULT_TRANSITION_RESULT
    )
    args = parser.parse_args(argv)
    payload = (
        register(args.output, args.transition_result)
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
