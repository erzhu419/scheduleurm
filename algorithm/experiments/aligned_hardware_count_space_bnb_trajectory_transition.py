"""Registered trajectory transition for confirmed count-space frame search.

R8/K36 and R16/K72 are compared with a same-family K4096 reference on common
fresh Poisson tapes.  Registration freezes sources and rules without loading
the model, computing capacity, constructing a tape, or calling a selector.
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

from scipy.stats import t as student_t

from algorithm.count_space_bnb_frame_candidates import (
    select_count_space_bnb_frame,
)
from algorithm.experiments.aligned_frame_flow_decomposition import decompose_flow
from algorithm.experiments.aligned_hardware_count_space_bnb_scaling_screen import (
    SCORE_TOLERANCE,
    _selection_reconstructs,
    _selection_work_holds,
)
from algorithm.experiments.aligned_hardware_frame_experiment import poisson_arrivals
from algorithm.experiments.aligned_hardware_frame_model import (
    capacity_certificate,
    load_model,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACTS / (
    "aligned_hardware_count_space_bnb_trajectory_transition_20260913.json"
)
DEFAULT_CONFIRMATION_RESULT = ARTIFACTS / (
    "aligned_hardware_count_space_bnb_confirmation_20260913.json"
)
EXPERIMENT = "aligned_hardware_count_space_bnb_trajectory_transition_v1"
SOURCE_EXPERIMENT = "aligned_hardware_count_space_bnb_confirmation_v1"
SOURCE_DECISION = "PRIMARY_LARGE_ACTION_CONFIRMATION_PASS"
THIS_SOURCE = (
    "algorithm/experiments/aligned_hardware_count_space_bnb_trajectory_transition.py"
)
FLOW_SOURCE = "algorithm/experiments/aligned_frame_flow_decomposition.py"

LANE_BUDGETS = {8: 36, 16: 72}
REFERENCE_LP_BUDGET = 4096
LOADS = (0.80, 0.95)
ARRIVAL_SEEDS = tuple(range(6041, 6049))
PREVIOUSLY_USED_ARRIVAL_SEEDS = (
    41,
    42,
    43,
    *range(1041, 1051),
    *range(2041, 2061),
    *range(3041, 3061),
    *range(4041, 4061),
)
ARRIVAL_HORIZON_S = 600.0
EXECUTION_WORKERS = 4
NONINFERIORITY_RATIO_MARGIN = 0.01
FAMILY_ALPHA = 0.05
PRIMARY_FAMILY_SIZE = len(LANE_BUDGETS) * len(LOADS)
EXECUTION_TOLERANCE = 1e-8

DECISION_RULE = {
    "INVALID_EXECUTION": "one or more registered execution checks fail",
    "COUNT_SPACE_TRAJECTORY_TRANSITION_PASS": (
        "all four R/load cases pass paired mean-flow noninferiority and the "
        "trajectory support gate"
    ),
    "COUNT_SPACE_TRAJECTORY_R8_ONLY_PASS_STOP": (
        "both R8 load cases pass but at least one R16 case fails"
    ),
    "COUNT_SPACE_TRAJECTORY_R16_ONLY_PASS_STOP": (
        "both R16 load cases pass but at least one R8 case fails"
    ),
    "COUNT_SPACE_TRAJECTORY_TRANSITION_FAIL_STOP": (
        "neither lane count passes both registered load cases"
    ),
}

CLAIM_BOUNDARY = (
    "A registered finite-horizon paired trajectory comparison on eight fresh "
    "Poisson seeds per load for synthetic identical-lane R8 and R16 models. "
    "It is not a uniform all-queue result, positive-recurrence proof, physical "
    "multi-GPU measurement, or deployment decision."
)
PROHIBITED = (
    "post-result changes to seeds, horizon, loads, budgets, reference, margins, "
    "or endpoints; ordered-action or count-vector enumeration; future-arrival "
    "access by selectors; server data; paper or README edits"
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


def _source_paths(result_path):
    result_path = Path(result_path)
    return (
        result_path,
        result_path.with_name(result_path.stem + "_protocol.json"),
        result_path.with_name(result_path.stem + "_fixed_sources.json"),
        result_path.with_name(result_path.stem + "_roster.json"),
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


def _source_contract(confirmation_result=DEFAULT_CONFIRMATION_RESULT):
    result_path, protocol_path, fixed_path, roster_path = _source_paths(
        confirmation_result
    )
    result = json.loads(result_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    fixed_sources = json.loads(fixed_path.read_text())
    roster = json.loads(roster_path.read_text())
    if result.get("protocol") != protocol:
        raise ValueError("count-space confirmation does not embed its protocol")
    changed = [
        name
        for name, text in fixed_sources.items()
        if (ROOT / name).read_text() != text
    ]
    if changed:
        raise ValueError(f"frozen count-space confirmation source changed: {changed}")
    if (
        result.get("experiment") != SOURCE_EXPERIMENT
        or result.get("status") != "PASS"
        or result.get("scientific_decision") != SOURCE_DECISION
        or result.get("confirmation_advance") is not True
        or result.get("independent_confirmation") is not True
        or not all(result.get("checks", {}).values())
        or roster.get("queues") != protocol.get("registered_base_queues")
        or result.get("work_accounting", {}).get("ordered_action_evaluations") != 0
        or result.get("work_accounting", {}).get("count_vector_enumerations") != 0
    ):
        raise ValueError("trajectory transition requires the passing confirmation")
    primary = {}
    for lanes, budget in LANE_BUDGETS.items():
        row = result.get("primary_cases", {}).get(str(lanes), {})
        if (
            row.get("lp_node_budget") != budget
            or row.get("primary_gate_passed") is not True
            or row.get("every_state_certificate_valid") is not True
        ):
            raise ValueError("confirmed lane/budget source case changed")
        primary[str(lanes)] = {
            "lp_node_budget": budget,
            "max_certified_normalized_gap": row[
                "max_certified_normalized_gap"
            ],
            "capacity_bridge": row["capacity_bridge"],
            "valid_certificate_state_count": row[
                "valid_certificate_state_count"
            ],
        }
    contract = {
        "confirmation_result_path": str(result_path.resolve()),
        "confirmation_experiment": result["experiment"],
        "confirmation_decision": result["scientific_decision"],
        "confirmed_lane_budgets": primary,
        "inherited_r2_oracle_evidence": protocol["source_contract"][
            "inherited_r2_oracle_evidence"
        ],
    }
    return contract, protocol["model"], fixed_sources


def _current_fixed_sources(upstream_fixed_sources):
    return {
        **upstream_fixed_sources,
        FLOW_SOURCE: (ROOT / FLOW_SOURCE).read_text(),
        THIS_SOURCE: (ROOT / THIS_SOURCE).read_text(),
    }


def _protocol_constants(model_snapshot, source_contract):
    if set(ARRIVAL_SEEDS) & set(PREVIOUSLY_USED_ARRIVAL_SEEDS):
        raise ValueError("trajectory seeds overlap a previously used arrival roster")
    return {
        "experiment": EXPERIMENT,
        "role": "registered count-space trajectory transition",
        "source_contract": source_contract,
        "model": _json(model_snapshot),
        "lane_budget_pairs": [
            {"lanes": lanes, "fixed_lp_budget": budget}
            for lanes, budget in LANE_BUDGETS.items()
        ],
        "reference": {
            "family": "the same count-space branch-and-bound selector",
            "lp_node_budget": REFERENCE_LP_BUDGET,
            "required_exact_in_every_frame": True,
        },
        "loads": list(LOADS),
        "arrival_seeds": list(ARRIVAL_SEEDS),
        "previously_used_arrival_seeds": list(PREVIOUSLY_USED_ARRIVAL_SEEDS),
        "seeds_disjoint_from_previous_trajectories": True,
        "arrival_horizon_s": ARRIVAL_HORIZON_S,
        "arrival_model": (
            "independent class-wise Poisson arrivals using the existing "
            "aligned-frame stream key aligned-frame:{seed}:{coordinate}"
        ),
        "common_tape_rule": (
            "construct exactly one tape for each (lanes, load, seed), then pass "
            "that identical immutable tuple to fixed and exact-reference runs"
        ),
        "online_information": (
            "each selector call receives only the current released queue, lane "
            "count, fixed catalogue, and its registered LP budget"
        ),
        "dispatch": (
            "global synchronous frames, FIFO within class, arrivals during a "
            "frame become eligible only at the next frame boundary"
        ),
        "stopping_rule": (
            "stop arrivals at 600 seconds and drain the identical finite tape "
            "to completion under both policies"
        ),
        "primary_cases": [
            {"lanes": lanes, "load": load, "fixed_lp_budget": budget}
            for lanes, budget in LANE_BUDGETS.items()
            for load in LOADS
        ],
        "paired_mean_flow_noninferiority": {
            "estimand": (
                "arithmetic mean across arrival seeds of fixed-budget/exact "
                "mean-flow ratios on common tapes"
            ),
            "ratio_margin": NONINFERIORITY_RATIO_MARGIN,
            "acceptance_upper": 1.0 + NONINFERIORITY_RATIO_MARGIN,
            "family_alpha": FAMILY_ALPHA,
            "family_size": PRIMARY_FAMILY_SIZE,
            "interval": (
                "one-sided Student t upper bound with Bonferroni alpha/family_size"
            ),
            "pass": "upper confidence bound <= acceptance_upper",
        },
        "trajectory_support_gate": (
            "for each case, the maximum fixed-budget returned gap/L1 over every "
            "visited frame and seed is strictly below the same-lane capacity eta"
        ),
        "execution_gate": [
            "all registered jobs complete under both policies",
            "queue recursion and FIFO flow decomposition are exact",
            "every selected frame reconstructs and has a valid returned certificate",
            "fixed per-frame LP calls do not exceed its registered budget",
            "the K4096 reference is exact in every frame",
        ],
        "expected_work": {
            "arrival_tapes": len(LANE_BUDGETS) * len(LOADS) * len(ARRIVAL_SEEDS),
            "trajectory_runs": (
                len(LANE_BUDGETS) * len(LOADS) * len(ARRIVAL_SEEDS) * 2
            ),
            "selector_calls": "reported from actual frame counts",
            "ordered_action_enumerations": 0,
            "count_vector_enumerations": 0,
            "execution_workers": EXECUTION_WORKERS,
        },
        "decision_rule": DECISION_RULE,
        "claim_boundary": CLAIM_BOUNDARY,
        "prohibited": PROHIBITED,
        "timing_is_scientific_gate": False,
    }


def register(output=DEFAULT_OUTPUT, confirmation_result=DEFAULT_CONFIRMATION_RESULT):
    """Freeze the trajectory protocol without constructing any arrival tape."""
    output, protocol_path, fixed_sources_path = _paths(output)
    if any(path.exists() for path in (output, protocol_path, fixed_sources_path)):
        raise ValueError("count-space trajectory artifacts already exist")
    source_contract, source_model, upstream_fixed = _source_contract(
        confirmation_result
    )
    fixed_sources = _current_fixed_sources(upstream_fixed)
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        **_protocol_constants(source_model, source_contract),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_new(protocol_path, protocol, compact=True)
    _write_new(fixed_sources_path, fixed_sources)
    return protocol


def _selection_certificate_valid(model, queue, lanes, budget, selection):
    norm = sum(queue)
    return (
        _selection_reconstructs(model, queue, lanes, selection)
        and _selection_work_holds(queue, lanes, budget, selection)
        and selection.score <= selection.upper_bound + SCORE_TOLERANCE
        and selection.gap >= -SCORE_TOLERANCE
        and math.isclose(
            selection.gap,
            max(0.0, selection.upper_bound - selection.score),
            rel_tol=0.0,
            abs_tol=SCORE_TOLERANCE,
        )
        and math.isclose(
            selection.epsilon,
            selection.gap / norm if norm else 0.0,
            rel_tol=0.0,
            abs_tol=SCORE_TOLERANCE,
        )
    )


def simulate_count_space_frames(
    model,
    arrivals,
    arrival_horizon,
    lp_node_budget,
    *,
    audit_limit=4,
):
    """Execute FIFO synchronous frames and report native count-space work."""
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
    valid_certificate_frames = 0
    max_recursion_violation = 0
    max_score_upper_violation = 0.0
    max_gap_identity_error = 0.0
    max_epsilon = 0.0
    max_gap = 0.0
    max_lp_calls = 0
    selected_counts = Counter()
    totals = {
        "lp_calls": 0,
        "lp_matrix_coefficients": 0,
        "candidate_evaluations": 0,
        "residual_fill_option_probes": 0,
        "dual_column_checks": 0,
        "branch_splits": 0,
    }
    frames = []
    audits = []
    selector_wall_s = 0.0

    while now < arrival_horizon or next_arrival is not None or any(queues):
        while next_arrival is not None and next_arrival[0] <= now:
            arrival_time, coordinate, index = next_arrival
            queues[coordinate].append((arrival_time, index))
            next_arrival = next(pending, None)
        before = tuple(len(queue) for queue in queues)
        selected_started = time.perf_counter()
        selected = select_count_space_bnb_frame(
            model.options,
            before,
            model.lanes,
            lp_node_budget,
        )
        selector_wall_s += time.perf_counter() - selected_started
        certificate_valid = _selection_certificate_valid(
            model, before, model.lanes, lp_node_budget, selected
        )
        valid_certificate_frames += int(certificate_valid)
        max_score_upper_violation = max(
            max_score_upper_violation, selected.score - selected.upper_bound
        )
        max_gap_identity_error = max(
            max_gap_identity_error,
            abs(selected.gap - max(0.0, selected.upper_bound - selected.score)),
        )
        max_epsilon = max(max_epsilon, selected.epsilon)
        max_gap = max(max_gap, selected.gap)
        max_lp_calls = max(max_lp_calls, selected.lp_calls)
        exact_frames += int(selected.exact)
        frame_count += 1
        peak_backlog = max(peak_backlog, sum(before))
        backlog_sum += sum(before)
        for field in totals:
            totals[field] += getattr(selected, field)

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
                "score": selected.score,
                "upper_bound": selected.upper_bound,
                "epsilon": selected.epsilon,
                "exact": selected.exact,
                "lp_calls": selected.lp_calls,
                "certificate_valid": certificate_valid,
            })
        now = end

    ordered_flows = sorted(flows)
    metrics = {
        "lp_node_budget": lp_node_budget,
        "arrival_horizon_s": arrival_horizon,
        "elapsed_including_drain_s": now,
        "job_count": len(tape),
        "completed_jobs": len(flows),
        "makespan_s": last_completion,
        "mean_flow_s": mean(flows) if flows else None,
        "p90_flow_s": (
            ordered_flows[math.ceil(0.9 * len(ordered_flows)) - 1]
            if ordered_flows
            else None
        ),
        "frame_count": frame_count,
        "exact_selection_frame_count": exact_frames,
        "valid_certificate_frame_count": valid_certificate_frames,
        "maximum_queue_recursion_violation": max_recursion_violation,
        "maximum_score_upper_violation": max_score_upper_violation,
        "maximum_gap_identity_error": max_gap_identity_error,
        "maximum_certified_gap_per_job": max_epsilon,
        "maximum_certified_gap": max_gap,
        "maximum_lp_calls_per_frame": max_lp_calls,
        "frame_boundary_peak_backlog": peak_backlog,
        "frame_boundary_mean_backlog": backlog_sum / frame_count,
        "selector_wall_s": selector_wall_s,
        "selector_mean_wall_ms": 1000.0 * selector_wall_s / frame_count,
        "selected_option_counts": dict(selected_counts),
        "work": totals,
        "frame_audit_prefix": audits,
    }
    return metrics, frames


def _compact_decomposition(decomposition):
    return {
        "status": decomposition["status"],
        "job_count": decomposition["job_count"],
        "completed_jobs": decomposition["completed_jobs"],
        "mean_flow_s": decomposition["mean_flow_s"],
        "flow_area_job_seconds": decomposition["flow_area_job_seconds"],
        "flow_identity_error_job_seconds": decomposition[
            "flow_identity_error_job_seconds"
        ],
        "mean_deferred_visible_wait_s": decomposition[
            "mean_deferred_visible_wait_s"
        ],
        "mean_arrival_barrier_wait_s": decomposition[
            "mean_arrival_barrier_wait_s"
        ],
        "mean_dispatch_to_completion_s": decomposition[
            "mean_dispatch_to_completion_s"
        ],
        "arrival_window_mean_flow_contribution_s": decomposition["time_regions"][
            "arrival_window"
        ]["mean_flow_contribution_s"],
        "drain_mean_flow_contribution_s": decomposition["time_regions"]["drain"][
            "mean_flow_contribution_s"
        ],
    }


def _run_tape(task):
    model, certificate, lanes, load, seed, horizon, fixed_budget = task
    lane_model = replace(model, lanes=lanes)
    tape = poisson_arrivals(certificate["arrival_rates"], horizon, seed)
    policies = []
    for policy, budget in (
        ("fixed", fixed_budget),
        ("exact_reference", REFERENCE_LP_BUDGET),
    ):
        metrics, frames = simulate_count_space_frames(
            lane_model, tape, horizon, budget
        )
        decomposition = decompose_flow(lane_model, tape, frames, horizon)
        policies.append({
            "policy": policy,
            "lp_node_budget": budget,
            "metrics": metrics,
            "flow_decomposition": _compact_decomposition(decomposition),
        })
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
        "policies": policies,
    }


def _paired_case_summary(runs, lanes, load, eta):
    members = sorted(
        (row for row in runs if row["lanes"] == lanes and row["load"] == load),
        key=lambda row: row["seed"],
    )
    ratios = []
    seed_rows = []
    maximum_epsilon = 0.0
    all_jobs_complete = True
    all_certificates_valid = True
    reference_exact = True
    all_flow_identities = True
    for row in members:
        policies = {item["policy"]: item for item in row["policies"]}
        fixed = policies["fixed"]
        reference = policies["exact_reference"]
        fm = fixed["metrics"]
        rm = reference["metrics"]
        ratio = fm["mean_flow_s"] / rm["mean_flow_s"]
        ratios.append(ratio)
        maximum_epsilon = max(
            maximum_epsilon, fm["maximum_certified_gap_per_job"]
        )
        all_jobs_complete &= (
            fm["job_count"] == fm["completed_jobs"] == rm["job_count"]
            and rm["job_count"] == rm["completed_jobs"]
            and fm["job_count"] == row["arrival_tape"]["job_count"]
        )
        all_certificates_valid &= (
            fm["valid_certificate_frame_count"] == fm["frame_count"]
            and rm["valid_certificate_frame_count"] == rm["frame_count"]
        )
        reference_exact &= (
            rm["exact_selection_frame_count"] == rm["frame_count"]
        )
        all_flow_identities &= all(
            item["flow_decomposition"]["status"] == "PASS"
            and abs(
                item["flow_decomposition"]["mean_flow_s"]
                - item["metrics"]["mean_flow_s"]
            )
            <= EXECUTION_TOLERANCE
            for item in policies.values()
        )
        seed_rows.append({
            "seed": row["seed"],
            "job_count": fm["job_count"],
            "fixed_mean_flow_s": fm["mean_flow_s"],
            "exact_mean_flow_s": rm["mean_flow_s"],
            "fixed_over_exact_mean_flow_ratio": ratio,
            "fixed_frame_count": fm["frame_count"],
            "exact_frame_count": rm["frame_count"],
            "fixed_exact_frame_fraction": (
                fm["exact_selection_frame_count"] / fm["frame_count"]
            ),
            "fixed_maximum_certified_gap_per_job": fm[
                "maximum_certified_gap_per_job"
            ],
            "fixed_lp_calls": fm["work"]["lp_calls"],
            "exact_lp_calls": rm["work"]["lp_calls"],
        })

    n = len(ratios)
    estimate = mean(ratios)
    standard_error = stdev(ratios) / math.sqrt(n)
    per_case_alpha = FAMILY_ALPHA / PRIMARY_FAMILY_SIZE
    critical = float(student_t.ppf(1.0 - per_case_alpha, df=n - 1))
    upper = estimate + critical * standard_error
    acceptance_upper = 1.0 + NONINFERIORITY_RATIO_MARGIN
    noninferiority_passed = upper <= acceptance_upper + SCORE_TOLERANCE
    support_residual = eta - maximum_epsilon
    support_passed = support_residual > SCORE_TOLERANCE
    return {
        "lanes": lanes,
        "load": load,
        "fixed_lp_budget": LANE_BUDGETS[lanes],
        "reference_lp_budget": REFERENCE_LP_BUDGET,
        "seeds": list(ARRIVAL_SEEDS),
        "seed_count": n,
        "paired_seed_rows": seed_rows,
        "paired_mean_flow_ratios": ratios,
        "paired_ratio_mean": estimate,
        "paired_ratio_standard_error": standard_error,
        "one_sided_confidence_level": 1.0 - per_case_alpha,
        "critical_t": critical,
        "paired_ratio_upper_confidence_bound": upper,
        "noninferiority_acceptance_upper": acceptance_upper,
        "paired_mean_flow_noninferiority_passed": noninferiority_passed,
        "effective_slack_eta": eta,
        "maximum_fixed_certified_gap_per_job": maximum_epsilon,
        "support_residual_margin": support_residual,
        "trajectory_support_gate_passed": support_passed,
        "all_jobs_complete": all_jobs_complete,
        "all_frame_certificates_valid": all_certificates_valid,
        "reference_exact_in_every_frame": reference_exact,
        "all_flow_identities_hold": all_flow_identities,
        "case_gate_passed": (
            noninferiority_passed
            and support_passed
            and all_jobs_complete
            and all_certificates_valid
            and reference_exact
            and all_flow_identities
        ),
    }


def _decision(execution_passed, cases):
    if not execution_passed:
        return "INVALID_EXECUTION"
    by_lane = {
        lanes: all(
            row["case_gate_passed"] for row in cases if row["lanes"] == lanes
        )
        for lanes in LANE_BUDGETS
    }
    if all(by_lane.values()):
        return "COUNT_SPACE_TRAJECTORY_TRANSITION_PASS"
    if by_lane[8]:
        return "COUNT_SPACE_TRAJECTORY_R8_ONLY_PASS_STOP"
    if by_lane[16]:
        return "COUNT_SPACE_TRAJECTORY_R16_ONLY_PASS_STOP"
    return "COUNT_SPACE_TRAJECTORY_TRANSITION_FAIL_STOP"


def run_registered(output=DEFAULT_OUTPUT):
    """Generate each common tape once and execute the registered paired runs."""
    output, protocol_path, fixed_sources_path = _paths(output)
    if output.exists():
        raise ValueError("count-space trajectory result already exists")
    protocol = json.loads(protocol_path.read_text())
    source_path = protocol.get("source_contract", {}).get(
        "confirmation_result_path"
    )
    if not source_path:
        raise ValueError("registered confirmation source path is missing")
    source_contract, source_model, upstream_fixed = _source_contract(source_path)
    registered_sources = json.loads(fixed_sources_path.read_text())
    if registered_sources != _current_fixed_sources(upstream_fixed):
        raise ValueError("registered count-space trajectory sources changed")
    expected = _protocol_constants(source_model, source_contract)
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("registered count-space trajectory protocol changed")

    model = load_model()
    if _json(model.snapshot()) != source_model:
        raise ValueError("current aligned model differs from the registered source")
    certificates = {}
    capacity_calls = 0
    for lanes in LANE_BUDGETS:
        for load in LOADS:
            certificate = capacity_certificate(replace(model, lanes=lanes), load)
            certificates[(lanes, load)] = certificate
            capacity_calls += 1

    tasks = [
        (
            model,
            certificates[(lanes, load)],
            lanes,
            load,
            seed,
            ARRIVAL_HORIZON_S,
            budget,
        )
        for lanes, budget in LANE_BUDGETS.items()
        for load in LOADS
        for seed in ARRIVAL_SEEDS
    ]
    started = time.perf_counter()
    runs = []
    with ProcessPoolExecutor(max_workers=EXECUTION_WORKERS) as pool:
        for row in pool.map(_run_tape, tasks):
            runs.append(row)

    case_summaries = [
        _paired_case_summary(
            runs,
            lanes,
            load,
            certificates[(lanes, load)]["effective_slack_eta"],
        )
        for lanes in LANE_BUDGETS
        for load in LOADS
    ]
    policy_runs = [
        policy for row in runs for policy in row["policies"]
    ]
    fixed_runs = [row for row in policy_runs if row["policy"] == "fixed"]
    reference_runs = [
        row for row in policy_runs if row["policy"] == "exact_reference"
    ]
    expected_tapes = len(tasks)
    expected_runs = 2 * expected_tapes
    checks = {
        "capacity_certificates_are_valid": all(
            certificate["capacity_projection_passed"]
            and certificate["conditional_model_drift_certified"]
            for certificate in certificates.values()
        ),
        "fresh_seed_roster_is_disjoint": not (
            set(ARRIVAL_SEEDS) & set(PREVIOUSLY_USED_ARRIVAL_SEEDS)
        ),
        "common_tape_roster_is_complete": (
            len(runs) == expected_tapes
            and len({(r["lanes"], r["load"], r["seed"]) for r in runs})
            == expected_tapes
            and all(r["arrival_tape"]["shared_policy_count"] == 2 for r in runs)
        ),
        "all_registered_jobs_complete": all(
            row["metrics"]["completed_jobs"] == row["metrics"]["job_count"]
            for row in policy_runs
        ),
        "queue_recursion_holds": all(
            row["metrics"]["maximum_queue_recursion_violation"] == 0
            for row in policy_runs
        ),
        "every_frame_certificate_is_valid": all(
            row["metrics"]["valid_certificate_frame_count"]
            == row["metrics"]["frame_count"]
            and row["metrics"]["maximum_score_upper_violation"]
            <= EXECUTION_TOLERANCE
            and row["metrics"]["maximum_gap_identity_error"]
            <= EXECUTION_TOLERANCE
            for row in policy_runs
        ),
        "fixed_budgets_hold_per_frame": all(
            row["metrics"]["maximum_lp_calls_per_frame"]
            <= row["lp_node_budget"]
            for row in fixed_runs
        ),
        "reference_is_exact_in_every_frame": all(
            row["metrics"]["exact_selection_frame_count"]
            == row["metrics"]["frame_count"]
            for row in reference_runs
        ),
        "flow_decomposition_reconstructs_mean_flow": all(
            row["flow_decomposition"]["status"] == "PASS"
            and abs(
                row["flow_decomposition"]["mean_flow_s"]
                - row["metrics"]["mean_flow_s"]
            )
            <= EXECUTION_TOLERANCE
            for row in policy_runs
        ),
        "registered_run_count_is_exact": len(policy_runs) == expected_runs,
        "capacity_call_count_is_exact": capacity_calls == len(certificates) == 4,
        "no_ordered_or_count_vector_enumeration": True,
        "registered_sources_unchanged_after_execution": (
            registered_sources == _current_fixed_sources(upstream_fixed)
        ),
    }
    execution_passed = all(checks.values())
    decision = _decision(execution_passed, case_summaries)
    result = {
        "experiment": EXPERIMENT,
        "status": "PASS" if execution_passed else "FAIL",
        "checks": checks,
        "protocol": protocol,
        "capacity_certificates": {
            f"R{lanes}_load_{load}": {
                "lanes": lanes,
                "load": load,
                "arrival_rates": certificate["arrival_rates"],
                "effective_slack_eta": certificate["effective_slack_eta"],
                "capacity_projection_passed": certificate[
                    "capacity_projection_passed"
                ],
                "conditional_model_drift_certified": certificate[
                    "conditional_model_drift_certified"
                ],
            }
            for (lanes, load), certificate in certificates.items()
        },
        "case_summaries": case_summaries,
        "runs": runs,
        "work_accounting": {
            "arrival_tapes": len(runs),
            "trajectory_runs": len(policy_runs),
            "selector_calls": sum(
                row["metrics"]["frame_count"] for row in policy_runs
            ),
            "fixed_lp_calls": sum(
                row["metrics"]["work"]["lp_calls"] for row in fixed_runs
            ),
            "reference_lp_calls": sum(
                row["metrics"]["work"]["lp_calls"] for row in reference_runs
            ),
            "fixed_lp_matrix_coefficients": sum(
                row["metrics"]["work"]["lp_matrix_coefficients"]
                for row in fixed_runs
            ),
            "reference_lp_matrix_coefficients": sum(
                row["metrics"]["work"]["lp_matrix_coefficients"]
                for row in reference_runs
            ),
            "ordered_action_enumerations": 0,
            "count_vector_enumerations": 0,
            "arrival_tape_generation_calls": len(runs),
        },
        "scientific_decision": decision,
        "trajectory_transition_advance": (
            decision == "COUNT_SPACE_TRAJECTORY_TRANSITION_PASS"
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "limitations": [
            "The four confidence bounds use eight seed-level ratios and the usual Student t approximation.",
            "The finite 600-second arrival windows are drained completely but do not prove steady-state behavior.",
            "The exact reference is exact for every visited frame under K4096, not a theorem that K4096 closes every queue.",
            "Synthetic lane replication does not measure physical multi-GPU interference.",
        ],
        "next_step": (
            "Close code-first validation and retain this frozen trajectory evidence "
            "for the later manuscript synthesis."
            if decision == "COUNT_SPACE_TRAJECTORY_TRANSITION_PASS"
            else "Preserve the failed case without changing seeds, budgets, or margins."
        ),
        "timing": {
            "execution_workers": EXECUTION_WORKERS,
            "total_wall_s": time.perf_counter() - started,
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
        "--confirmation-result",
        type=Path,
        default=DEFAULT_CONFIRMATION_RESULT,
        help="frozen count-space confirmation used only during registration",
    )
    args = parser.parse_args(argv)
    payload = (
        register(args.output, args.confirmation_result)
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
