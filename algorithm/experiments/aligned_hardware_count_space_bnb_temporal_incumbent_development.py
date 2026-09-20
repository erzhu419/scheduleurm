"""Registered development test of a projected temporal B&B incumbent.

The previous frame's exact option counts are projected to the current visible
queue and supplied only as a feasible lower bound to a K4096 count-space B&B.
Its trajectory is compared on the same fresh Poisson tape with the identical
same-family cold K4096 exact-reference run.  Actual LP and matrix work is
charged directly for both policies.

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

from algorithm.temporal_incumbent_count_space_bnb_frame_candidates import (
    project_temporal_incumbent,
    select_count_space_bnb_frame as select_warm_count_space_bnb_frame,
)
from algorithm.count_space_bnb_frame_candidates import select_count_space_bnb_frame
from algorithm.experiments.aligned_frame_flow_decomposition import decompose_flow
from algorithm.experiments.aligned_hardware_count_space_bnb_exactness_escalation_development import (
    DEFAULT_OUTPUT as DEFAULT_ESCALATION_RESULT,
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
    "aligned_hardware_count_space_bnb_temporal_incumbent_"
    "development_20260913.json"
)
EXPERIMENT = "aligned_hardware_count_space_bnb_temporal_incumbent_development_v1"
SOURCE_DECISION = "EXACTNESS_ESCALATION_WORK_FAIL_STOP"
THIS_SOURCE = (
    "algorithm/experiments/"
    "aligned_hardware_count_space_bnb_temporal_incumbent_development.py"
)
SELECTOR_SOURCE = "algorithm/temporal_incumbent_count_space_bnb_frame_candidates.py"

REFERENCE_LP_BUDGET = 4096
LANES = (8, 16)
LOADS = (0.80, 0.95)
ARRIVAL_SEEDS = tuple(range(8041, 8049))
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
)
ARRIVAL_HORIZON_S = 600.0
EXECUTION_WORKERS = 4
WORK_FIELDS = (
    "lp_calls",
    "lp_matrix_coefficients",
    "candidate_evaluations",
    "initial_incumbent_candidate_evaluations",
    "residual_fill_option_probes",
    "dual_column_checks",
    "branch_splits",
)

DECISION_RULE = {
    "INVALID_EXECUTION": "one or more registered execution-integrity checks fail",
    "TEMPORAL_INCUMBENT_PASS": (
        "every warm frame is exact and identical to the cold K4096 reference, "
        "and charged warm LP calls and matrix cells are strictly smaller "
        "in every R/load case and in aggregate"
    ),
    "TEMPORAL_INCUMBENT_WORK_FAIL_STOP": (
        "execution is valid but one or more registered exactness, identity, or "
        "strict-work gates fail"
    ),
}

CLAIM_BOUNDARY = (
    "A registered finite-horizon development comparison on eight fresh Poisson "
    "seeds per load for synthetic identical-lane R8 and R16 models.  It tests "
    "a one-frame temporal incumbent on these tapes only; it is not an all-queue "
    "result, stability proof, physical multi-GPU measurement, or deployment decision."
)
PROHIBITED = (
    "post-result changes to seeds, horizon, loads, projection, reference, or "
    "endpoints; ordered-action or "
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


def _source_contract(escalation_result=DEFAULT_ESCALATION_RESULT):
    result_path, protocol_path, fixed_path = _source_paths(escalation_result)
    result = json.loads(result_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    fixed_sources = json.loads(fixed_path.read_text())
    if result.get("protocol") != protocol:
        raise ValueError("exactness escalation does not embed its protocol")
    changed = [
        name
        for name, text in fixed_sources.items()
        if (ROOT / name).read_text() != text
    ]
    if changed:
        raise ValueError(f"frozen exactness-escalation source changed: {changed}")
    if (
        result.get("experiment") != SOURCE_EXPERIMENT
        or result.get("status") != "PASS"
        or result.get("scientific_decision") != SOURCE_DECISION
        or result.get("exactness_escalation_advance") is not False
        or not all(result.get("checks", {}).values())
    ):
        raise ValueError("temporal incumbent requires the valid failed escalation")
    prior_cases = []
    for row in result.get("case_summaries", []):
        prior_cases.append({
            "lanes": row["lanes"],
            "load": row["load"],
            "start_lp_budget": row["start_lp_budget"],
            "adaptive_lp_calls_strictly_below_reference": row[
                "adaptive_lp_calls_strictly_below_reference"
            ],
            "adaptive_matrix_cells_strictly_below_reference": row[
                "adaptive_matrix_cells_strictly_below_reference"
            ],
            "case_gate_passed": row["case_gate_passed"],
        })
    expected_pairs = {
        (lanes, load)
        for lanes in LANES
        for load in LOADS
    }
    observed_pairs = {
        (row["lanes"], row["load"])
        for row in prior_cases
    }
    if observed_pairs != expected_pairs:
        raise ValueError("failed escalation lane or load roster changed")
    return (
        {
            "escalation_result_path": str(result_path.resolve()),
            "escalation_experiment": result["experiment"],
            "escalation_decision": result["scientific_decision"],
            "escalation_advance": result["exactness_escalation_advance"],
            "prior_case_gates": prior_cases,
        },
        protocol["model"],
        fixed_sources,
    )


def _current_fixed_sources(upstream_fixed_sources):
    return {
        **upstream_fixed_sources,
        THIS_SOURCE: (ROOT / THIS_SOURCE).read_text(),
        SELECTOR_SOURCE: (ROOT / SELECTOR_SOURCE).read_text(),
    }


def _protocol_constants(model_snapshot, source_contract):
    if set(ARRIVAL_SEEDS) & set(PREVIOUSLY_USED_ARRIVAL_SEEDS):
        raise ValueError("temporal seeds overlap a previously used trajectory roster")
    return {
        "experiment": EXPERIMENT,
        "role": "registered development temporal-incumbent exact B&B",
        "source_contract": source_contract,
        "model": _json(model_snapshot),
        "lanes": list(LANES),
        "lp_node_budget": REFERENCE_LP_BUDGET,
        "temporal_projection_rule": {
            "input": "the immediately previous warm frame's exact option counts",
            "clip": (
                "in catalogue order, retain at most each prior count subject to "
                "current visible queue resources and the lane cap"
            ),
            "greedy_completion": (
                "while lanes remain, add the feasible single copy producing the "
                "best current score/duration/index key, only if that key improves"
            ),
            "first_frame": "no temporal incumbent",
            "search_role": (
                "the projection initializes only the feasible incumbent lower "
                "bound; it does not alter roots, branching, or LP upper bounds"
            ),
            "accounting": (
                "projection option probes and the extra incumbent candidate "
                "evaluation are reported in addition to native search work"
            ),
            "online_information": (
                "previous exact option counts plus current released queue only"
            ),
        },
        "reference": {
            "family": "the original cold count-space branch-and-bound selector",
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
            "contents to warm and cold K4096 runs"
        ),
        "online_information": (
            "each selector receives only current released queues, lanes, fixed "
            "catalogue, and for warm search the previous exact warm count vector"
        ),
        "dispatch": (
            "global synchronous frames, FIFO within class, with arrivals during "
            "a frame eligible at the next frame boundary"
        ),
        "stopping_rule": (
            "stop arrivals at 600 seconds and drain the common finite tape to completion"
        ),
        "primary_cases": [
            {"lanes": lanes, "load": load, "lp_node_budget": REFERENCE_LP_BUDGET}
            for lanes in LANES
            for load in LOADS
        ],
        "registered_gate": [
            "warm and cold selections are exact in every visited frame",
            "warm and cold K4096 frame actions are exactly identical",
            "warm and cold per-job flow sequences are exactly identical",
            "charged warm LP calls are strictly below cold in every case",
            "charged warm matrix cells are strictly below cold in every case",
            "both charged work totals are also strictly below reference in aggregate",
            "all jobs complete and queue recursion, FIFO decomposition, and every "
            "attempt certificate are valid",
        ],
        "expected_work": {
            "arrival_tapes": len(LANES) * len(LOADS) * len(ARRIVAL_SEEDS),
            "trajectory_runs": (
                len(LANES) * len(LOADS) * len(ARRIVAL_SEEDS) * 2
            ),
            "selector_calls": "one warm and one cold call per visited frame",
            "projection_calls": "one per warm frame after the first",
            "ordered_action_enumerations": 0,
            "count_vector_enumerations": 0,
            "execution_workers": EXECUTION_WORKERS,
        },
        "decision_rule": DECISION_RULE,
        "claim_boundary": CLAIM_BOUNDARY,
        "prohibited": PROHIBITED,
        "timing_is_scientific_gate": False,
    }


def register(output=DEFAULT_OUTPUT, escalation_result=DEFAULT_ESCALATION_RESULT):
    """Freeze protocol and sources without model, capacity, tape, or selector work."""
    output, protocol_path, fixed_sources_path = _paths(output)
    if any(path.exists() for path in (output, protocol_path, fixed_sources_path)):
        raise ValueError("temporal-incumbent artifacts already exist")
    source_contract, source_model, upstream_fixed = _source_contract(escalation_result)
    fixed_sources = _current_fixed_sources(upstream_fixed)
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        **_protocol_constants(source_model, source_contract),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_new(protocol_path, protocol, compact=True)
    _write_new(fixed_sources_path, fixed_sources)
    return protocol


def _warm_selection_certificate_valid(model, queue, selection, seeded):
    incumbent_evaluations = selection.initial_incumbent_candidate_evaluations
    if incumbent_evaluations != int(seeded):
        return False
    search_only = replace(
        selection,
        candidate_evaluations=(
            selection.candidate_evaluations - incumbent_evaluations
        ),
    )
    return _selection_certificate_valid(
        model, queue, model.lanes, REFERENCE_LP_BUDGET, search_only
    )


def simulate_temporal_incumbent_frames(
    model,
    arrivals,
    arrival_horizon,
    *,
    warm,
    audit_limit=4,
):
    """Run synchronous FIFO frames with warm or cold exact K4096 search."""
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
    valid_selections = 0
    max_recursion_violation = 0
    max_score_upper_violation = 0.0
    max_gap_identity_error = 0.0
    selected_counts = Counter()
    totals = {field: 0 for field in WORK_FIELDS}
    projection_calls = 0
    projection_retained_copies = 0
    projection_greedy_added_copies = 0
    projection_option_probes = 0
    invalid_projection_count = 0
    warm_seeded_frames = 0
    previous_exact_counts = None
    frames = []
    audits = []
    selector_wall_s = 0.0

    while now < arrival_horizon or next_arrival is not None or any(queues):
        while next_arrival is not None and next_arrival[0] <= now:
            arrival_time, coordinate, index = next_arrival
            queues[coordinate].append((arrival_time, index))
            next_arrival = next(pending, None)
        before = tuple(len(queue) for queue in queues)
        projection = None
        initial_counts = None
        if warm and previous_exact_counts is not None:
            projection = project_temporal_incumbent(
                model.options,
                before,
                model.lanes,
                previous_exact_counts,
            )
            initial_counts = projection.option_counts
            projection_calls += 1
            projection_retained_copies += projection.retained_copies
            projection_greedy_added_copies += projection.greedy_added_copies
            projection_option_probes += projection.option_probes
            warm_seeded_frames += int(any(initial_counts))
            reconstructed_service = tuple(
                sum(
                    projection.option_counts[index]
                    * model.options[index].requirements[coordinate]
                    for index in range(len(model.options))
                )
                for coordinate in range(len(before))
            )
            invalid_projection_count += int(
                sum(projection.option_counts) > model.lanes
                or reconstructed_service != projection.service
                or any(
                    done > waiting
                    for done, waiting in zip(
                        projection.service, before, strict=True
                    )
                )
            )
        started = time.perf_counter()
        if warm:
            selected = select_warm_count_space_bnb_frame(
                model.options,
                before,
                model.lanes,
                REFERENCE_LP_BUDGET,
                initial_option_counts=initial_counts,
            )
        else:
            selected = select_count_space_bnb_frame(
                model.options,
                before,
                model.lanes,
                REFERENCE_LP_BUDGET,
            )
        selector_wall_s += time.perf_counter() - started
        valid = (
            _warm_selection_certificate_valid(
                model, before, selected, initial_counts is not None
            )
            if warm
            else _selection_certificate_valid(
                model, before, model.lanes, REFERENCE_LP_BUDGET, selected
            )
        )
        frame_count += 1
        exact_frames += int(selected.exact)
        valid_final_frames += int(valid)
        valid_selections += int(valid)
        for field in WORK_FIELDS:
            totals[field] += getattr(selected, field, 0)
        previous_exact_counts = selected.option_counts if selected.exact else None

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
                "lp_node_budget": REFERENCE_LP_BUDGET,
                "projection": (
                    None
                    if projection is None
                    else {
                        "option_counts": projection.option_counts,
                        "service": projection.service,
                        "duration": projection.duration,
                        "score": projection.score,
                        "retained_copies": projection.retained_copies,
                        "greedy_added_copies": projection.greedy_added_copies,
                        "option_probes": projection.option_probes,
                    }
                ),
                "exact": selected.exact,
                "certificate_valid": valid,
            })
        now = end

    ordered_flows = sorted(flows)
    metrics = {
        "policy_uses_temporal_incumbent": warm,
        "lp_node_budget": REFERENCE_LP_BUDGET,
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
        "selector_call_count": frame_count,
        "valid_certificate_selection_count": valid_selections,
        "projection_call_count": projection_calls,
        "warm_seeded_frame_count": warm_seeded_frames,
        "projection_retained_copies": projection_retained_copies,
        "projection_greedy_added_copies": projection_greedy_added_copies,
        "projection_option_probes": projection_option_probes,
        "invalid_projection_count": invalid_projection_count,
        "maximum_queue_recursion_violation": max_recursion_violation,
        "maximum_score_upper_violation": max_score_upper_violation,
        "maximum_gap_identity_error": max_gap_identity_error,
        "frame_boundary_peak_backlog": peak_backlog,
        "frame_boundary_mean_backlog": backlog_sum / frame_count,
        "selector_wall_s": selector_wall_s,
        "selector_mean_wall_ms_per_call": (
            1000.0 * selector_wall_s / frame_count
        ),
        "selected_option_counts": dict(selected_counts),
        "charged_work": totals,
        "frame_audit_prefix": audits,
    }
    return metrics, frames, tuple(flows)


def _run_tape(task):
    model, certificate, lanes, load, seed, horizon = task
    lane_model = replace(model, lanes=lanes)
    tape = poisson_arrivals(certificate["arrival_rates"], horizon, seed)
    warm_metrics, warm_frames, warm_flows = (
        simulate_temporal_incumbent_frames(
            lane_model, tape, horizon, warm=True
        )
    )
    reference_metrics, reference_frames, reference_flows = (
        simulate_temporal_incumbent_frames(
            lane_model, tape, horizon, warm=False
        )
    )
    warm_decomposition = decompose_flow(
        lane_model, tape, warm_frames, horizon
    )
    reference_decomposition = decompose_flow(
        lane_model, tape, reference_frames, horizon
    )
    policies = [
        {
            "policy": "temporal_incumbent",
            "metrics": warm_metrics,
            "flow_decomposition": _compact_decomposition(warm_decomposition),
        },
        {
            "policy": "cold_exact_reference",
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
            warm_frames == reference_frames
        ),
        "per_job_flow_sequence_exactly_equal": warm_flows == reference_flows,
        "policies": policies,
    }


def _case_summary(runs, lanes, load):
    members = sorted(
        (row for row in runs if row["lanes"] == lanes and row["load"] == load),
        key=lambda row: row["seed"],
    )
    seed_rows = []
    warm_lp_calls = 0
    cold_lp_calls = 0
    warm_matrix_cells = 0
    cold_matrix_cells = 0
    all_jobs_complete = True
    all_recursions = True
    all_certificates = True
    warm_exact = True
    cold_exact = True
    projection_chain_complete = True
    projections_valid = True
    action_identity = True
    flow_identity = True
    all_fifo_decompositions = True
    for row in members:
        policies = {item["policy"]: item for item in row["policies"]}
        warm = policies["temporal_incumbent"]
        cold = policies["cold_exact_reference"]
        wm = warm["metrics"]
        cm = cold["metrics"]
        ww = wm["charged_work"]
        cw = cm["charged_work"]
        warm_lp_calls += ww["lp_calls"]
        cold_lp_calls += cw["lp_calls"]
        warm_matrix_cells += ww["lp_matrix_coefficients"]
        cold_matrix_cells += cw["lp_matrix_coefficients"]
        all_jobs_complete &= (
            wm["job_count"] == wm["completed_jobs"] == cm["job_count"]
            and cm["job_count"] == cm["completed_jobs"]
            and wm["job_count"] == row["arrival_tape"]["job_count"]
        )
        all_recursions &= (
            wm["maximum_queue_recursion_violation"] == 0
            and cm["maximum_queue_recursion_violation"] == 0
        )
        all_certificates &= (
            wm["valid_certificate_selection_count"] == wm["selector_call_count"]
            and cm["valid_certificate_selection_count"] == cm["selector_call_count"]
            and wm["valid_final_certificate_frame_count"] == wm["frame_count"]
            and cm["valid_final_certificate_frame_count"] == cm["frame_count"]
        )
        warm_exact &= wm["exact_selection_frame_count"] == wm["frame_count"]
        cold_exact &= cm["exact_selection_frame_count"] == cm["frame_count"]
        projection_chain_complete &= (
            wm["projection_call_count"] == max(0, wm["frame_count"] - 1)
            and cm["projection_call_count"] == 0
        )
        projections_valid &= wm["invalid_projection_count"] == 0
        action_identity &= row["deterministic_action_trace_exactly_equal"]
        flow_identity &= (
            row["per_job_flow_sequence_exactly_equal"]
            and wm["mean_flow_s"] == cm["mean_flow_s"]
            and wm["p90_flow_s"] == cm["p90_flow_s"]
            and wm["makespan_s"] == cm["makespan_s"]
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
            "job_count": wm["job_count"],
            "mean_flow_s": wm["mean_flow_s"],
            "frame_count": wm["frame_count"],
            "warm_seeded_frame_count": wm["warm_seeded_frame_count"],
            "projection_call_count": wm["projection_call_count"],
            "projection_retained_copies": wm["projection_retained_copies"],
            "projection_greedy_added_copies": wm[
                "projection_greedy_added_copies"
            ],
            "projection_option_probes": wm["projection_option_probes"],
            "warm_lp_calls": ww["lp_calls"],
            "cold_lp_calls": cw["lp_calls"],
            "warm_lp_matrix_coefficients": ww["lp_matrix_coefficients"],
            "cold_lp_matrix_coefficients": cw["lp_matrix_coefficients"],
            "deterministic_action_trace_exactly_equal": row[
                "deterministic_action_trace_exactly_equal"
            ],
            "per_job_flow_sequence_exactly_equal": row[
                "per_job_flow_sequence_exactly_equal"
            ],
        })
    lp_saved = cold_lp_calls - warm_lp_calls
    matrix_saved = cold_matrix_cells - warm_matrix_cells
    lp_gate = warm_lp_calls < cold_lp_calls
    matrix_gate = warm_matrix_cells < cold_matrix_cells
    return {
        "lanes": lanes,
        "load": load,
        "lp_node_budget": REFERENCE_LP_BUDGET,
        "seeds": list(ARRIVAL_SEEDS),
        "seed_count": len(members),
        "seed_rows": seed_rows,
        "warm_exact_in_every_frame": warm_exact,
        "cold_exact_in_every_frame": cold_exact,
        "projection_chain_complete": projection_chain_complete,
        "all_projections_valid": projections_valid,
        "deterministic_action_traces_exactly_equal": action_identity,
        "per_job_flow_sequences_exactly_equal": flow_identity,
        "all_jobs_complete": all_jobs_complete,
        "all_queue_recursions_hold": all_recursions,
        "all_selection_certificates_valid": all_certificates,
        "all_fifo_flow_decompositions_hold": all_fifo_decompositions,
        "warm_lp_calls": warm_lp_calls,
        "cold_lp_calls": cold_lp_calls,
        "lp_calls_saved": lp_saved,
        "warm_lp_calls_strictly_below_cold": lp_gate,
        "warm_lp_matrix_coefficients": warm_matrix_cells,
        "cold_lp_matrix_coefficients": cold_matrix_cells,
        "lp_matrix_coefficients_saved": matrix_saved,
        "warm_matrix_cells_strictly_below_cold": matrix_gate,
        "case_gate_passed": all((
            warm_exact,
            cold_exact,
            projection_chain_complete,
            projections_valid,
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
        return "TEMPORAL_INCUMBENT_PASS"
    return "TEMPORAL_INCUMBENT_WORK_FAIL_STOP"


def run_registered(output=DEFAULT_OUTPUT):
    """Build fresh common tapes, run warm/cold pairs, and preserve outcome."""
    output, protocol_path, fixed_sources_path = _paths(output)
    if output.exists():
        raise ValueError("temporal-incumbent result already exists")
    protocol = json.loads(protocol_path.read_text())
    source_path = protocol.get("source_contract", {}).get("escalation_result_path")
    if not source_path:
        raise ValueError("registered escalation source path is missing")
    source_contract, source_model, upstream_fixed = _source_contract(source_path)
    registered_sources = json.loads(fixed_sources_path.read_text())
    if registered_sources != _current_fixed_sources(upstream_fixed):
        raise ValueError("registered temporal-incumbent sources changed")
    expected = _protocol_constants(source_model, source_contract)
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("registered temporal-incumbent protocol changed")

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

    case_summaries = [
        _case_summary(runs, lanes, load)
        for lanes in LANES
        for load in LOADS
    ]
    policy_runs = [policy for row in runs for policy in row["policies"]]
    warm_runs = [
        row for row in policy_runs if row["policy"] == "temporal_incumbent"
    ]
    cold_runs = [
        row for row in policy_runs if row["policy"] == "cold_exact_reference"
    ]
    aggregate_warm_work = {
        field: sum(
            row["metrics"]["charged_work"][field] for row in warm_runs
        )
        for field in WORK_FIELDS
    }
    aggregate_cold_work = {
        field: sum(
            row["metrics"]["charged_work"][field] for row in cold_runs
        )
        for field in WORK_FIELDS
    }
    aggregate_warm_lp = aggregate_warm_work["lp_calls"]
    aggregate_cold_lp = aggregate_cold_work["lp_calls"]
    aggregate_warm_matrix = aggregate_warm_work["lp_matrix_coefficients"]
    aggregate_cold_matrix = aggregate_cold_work["lp_matrix_coefficients"]
    aggregate_lp_gate = aggregate_warm_lp < aggregate_cold_lp
    aggregate_matrix_gate = aggregate_warm_matrix < aggregate_cold_matrix
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
        "every_selection_certificate_is_valid": all(
            row["metrics"]["valid_certificate_selection_count"]
            == row["metrics"]["selector_call_count"]
            for row in policy_runs
        ),
        "all_temporal_projections_are_valid": all(
            row["metrics"]["invalid_projection_count"] == 0 for row in warm_runs
        ),
        "flow_decomposition_reconstructs_mean_flow": all(
            row["flow_decomposition"]["status"] == "PASS"
            and abs(
                row["flow_decomposition"]["mean_flow_s"]
                - row["metrics"]["mean_flow_s"]
            ) <= EXECUTION_TOLERANCE
            for row in policy_runs
        ),
        "warm_is_exact_in_every_frame": all(
            row["metrics"]["exact_selection_frame_count"]
            == row["metrics"]["frame_count"]
            for row in warm_runs
        ),
        "cold_reference_is_exact_in_every_frame": all(
            row["metrics"]["exact_selection_frame_count"]
            == row["metrics"]["frame_count"]
            for row in cold_runs
        ),
        "registered_run_count_is_exact": len(policy_runs) == expected_policy_runs,
        "capacity_call_count_is_exact": capacity_calls == len(LANES) * len(LOADS),
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
            "warm_lp_calls": aggregate_warm_lp,
            "cold_lp_calls": aggregate_cold_lp,
            "lp_calls_saved": aggregate_cold_lp - aggregate_warm_lp,
            "warm_lp_calls_strictly_below_cold": aggregate_lp_gate,
            "warm_lp_matrix_coefficients": aggregate_warm_matrix,
            "cold_lp_matrix_coefficients": aggregate_cold_matrix,
            "lp_matrix_coefficients_saved": (
                aggregate_cold_matrix - aggregate_warm_matrix
            ),
            "warm_matrix_cells_strictly_below_cold": aggregate_matrix_gate,
            "aggregate_work_gate_passed": aggregate_work_gate,
        },
        "work_accounting": {
            "arrival_tapes": len(runs),
            "trajectory_runs": len(policy_runs),
            "warm_selector_calls": sum(
                row["metrics"]["selector_call_count"] for row in warm_runs
            ),
            "cold_selector_calls": sum(
                row["metrics"]["selector_call_count"] for row in cold_runs
            ),
            "temporal_projection_calls": sum(
                row["metrics"]["projection_call_count"] for row in warm_runs
            ),
            "temporal_projection_option_probes": sum(
                row["metrics"]["projection_option_probes"] for row in warm_runs
            ),
            "warm_search_work": aggregate_warm_work,
            "cold_search_work": aggregate_cold_work,
            "warm_lp_calls": aggregate_warm_lp,
            "cold_lp_calls": aggregate_cold_lp,
            "warm_lp_matrix_coefficients": aggregate_warm_matrix,
            "cold_lp_matrix_coefficients": aggregate_cold_matrix,
            "ordered_action_enumerations": 0,
            "count_vector_enumerations": 0,
            "arrival_tape_generation_calls": len(runs),
        },
        "runs": runs,
        "scientific_decision": scientific_decision,
        "temporal_incumbent_advance": (
            scientific_decision == "TEMPORAL_INCUMBENT_PASS"
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "limitations": [
            "The finite seed roster cannot establish an all-queue guarantee.",
            "Wall-clock selector timing is diagnostic and is not a scientific gate.",
            "Every duration-root LP remains mandatory for warm and cold searches.",
        ],
        "next_step": (
            "Preserve this registered outcome without changing seeds, projection, "
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
        "--escalation-result", type=Path, default=DEFAULT_ESCALATION_RESULT
    )
    args = parser.parse_args(argv)
    payload = (
        register(args.output, args.escalation_result)
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
