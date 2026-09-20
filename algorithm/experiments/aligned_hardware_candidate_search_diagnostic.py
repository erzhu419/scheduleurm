"""Trace the frozen v4 candidate search on its complete development state roster.

This is an offline mechanism diagnostic.  It replays the already consumed v4
K26 and K120 trajectories only to recover every visited queue.  Each distinct
queue then receives one deterministic K120 search trace, from which the K26
prefix is recovered.  No traced decision is fed back into either trajectory.

The diagnostic separates the K26 certified gap as

    U26 - S26 = (S* - S26) + (U26 - S*),

where the two terms are finite-budget support regret and frontier-bound slack.  It
also reports the paid attempt at which the final exact score and action first
appear.  Run registration before execution so the roster and source text are
fixed before any queue states are observed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from algorithm.budgeted_frame_candidates import select_budgeted_frame
from algorithm.experiments.aligned_budgeted_frame_search_trace import (
    trace_budgeted_frame_search,
)
from algorithm.experiments.aligned_hardware_frame_confirmation import _write_new
from algorithm.experiments.aligned_hardware_frame_experiment import (
    exact_search_budget,
    poisson_arrivals,
    simulate_frames,
)
from algorithm.experiments.aligned_hardware_frame_model import (
    capacity_certificate,
    load_model,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACTS / "aligned_hardware_candidate_search_diagnostic_development_20260910.json"
DEVELOPMENT = ARTIFACTS / "aligned_hardware_frame_experiment_v4_20260909.json"
V4_SOURCE_SNAPSHOT = ARTIFACTS / "aligned_hardware_frame_confirmation_v4_20260909/source_snapshot.json"
LOADS = (0.50, 0.80, 0.95)
SEEDS = (41, 42, 43)
SMALL_BUDGET = 26
EXACT_BUDGET = 120
HORIZON = 20_000.0
SCORE_TOLERANCE = 1e-10
FROZEN_V4_SOURCES = (
    "algorithm/budgeted_frame_candidates.py",
    "algorithm/experiments/aligned_hardware_frame_model.py",
    "algorithm/experiments/aligned_hardware_frame_experiment.py",
)
NEW_SOURCES = (
    "algorithm/experiments/aligned_budgeted_frame_search_trace.py",
    "algorithm/experiments/aligned_hardware_candidate_search_diagnostic.py",
)


def _json(value):
    return json.loads(json.dumps(value))


def _registered_rows(development):
    rows = [
        row for row in development["runs"]
        if row["load"] in LOADS and row["seed"] in SEEDS
        and row["budget"] in (SMALL_BUDGET, EXACT_BUDGET)
    ]
    rows.sort(key=lambda row: (row["load"], row["seed"], row["budget"]))
    return _json(rows)


def _source_texts():
    frozen = json.loads(V4_SOURCE_SNAPSHOT.read_text())
    sources = {}
    for name in FROZEN_V4_SOURCES:
        current = (ROOT / name).read_text()
        if frozen.get(name) != current:
            raise ValueError(f"frozen v4 source changed: {name}")
        sources[name] = current
    sources.update({name: (ROOT / name).read_text() for name in NEW_SOURCES})
    return sources


def _validate_development(development):
    protocol = development["protocol"]
    if (development["experiment"] != "aligned_hardware_frame_and_budget_v4"
            or development["status"] != "PASS"
            or not all(development["checks"].values())
            or protocol["loads"] != list(LOADS)
            or protocol["seeds"] != list(SEEDS)
            or protocol["arrival_horizon_s"] != HORIZON
            or protocol["exact_reference_budget"] != EXACT_BUDGET
            or protocol["budgets"] != [10, 14, SMALL_BUDGET, EXACT_BUDGET]):
        raise ValueError("candidate-search diagnosis requires the frozen v4 development roster")
    model = load_model()
    if (_json(model.snapshot()) != development["model"]
            or exact_search_budget(len(model.options), model.lanes) != EXACT_BUDGET):
        raise ValueError("candidate-search model or exact budget changed")
    certificates = [capacity_certificate(model, load) for load in LOADS]
    if _json(certificates) != development["certificates"]:
        raise ValueError("candidate-search capacity certificates changed")
    rows = _registered_rows(development)
    expected_roster = [
        (load, seed, budget)
        for load in LOADS for seed in SEEDS
        for budget in (SMALL_BUDGET, EXACT_BUDGET)
    ]
    if ([(row["load"], row["seed"], row["budget"]) for row in rows] != expected_roster
            or any(row["maximum_candidate_attempts_per_frame"] > row["budget"] for row in rows)
            or any(row["exact_selection_frame_count"] != row["frame_count"]
                   for row in rows if row["budget"] == EXACT_BUDGET)):
        raise ValueError("candidate-search source rows do not match the 18-run roster")
    return model, certificates, rows


class QueueRecorder:
    """Observe queue visits while returning the unmodified frozen v4 decision."""

    def __init__(self, model, budget):
        self.model = model
        self.budget = budget
        self.visits = Counter()
        self.duration_by_queue = Counter()

    def __call__(self, options, queues, lanes, budget):
        if tuple(options) != self.model.options or lanes != self.model.lanes or budget != self.budget:
            raise ValueError("queue recorder received a selector call outside its source trajectory")
        queue = tuple(queues)
        selected = select_budgeted_frame(options, queue, lanes, budget)
        self.visits[queue] += 1
        self.duration_by_queue[queue] += selected.duration
        return selected


def _changed_metric_fields(metrics, source_row):
    expected = {key: value for key, value in source_row.items() if key not in ("load", "seed")}
    observed = _json(metrics)
    return sorted(key for key, value in expected.items() if observed.get(key) != value)


def _replay_source(model, certificate, source_row, tape):
    recorder = QueueRecorder(model, source_row["budget"])
    metrics = simulate_frames(
        model, tape, HORIZON, certificate, source_row["budget"], selector=recorder,
    )
    changed = _changed_metric_fields(metrics, source_row)
    if changed:
        raise ValueError(
            f"frozen trajectory changed at load={source_row['load']}, "
            f"seed={source_row['seed']}, budget={source_row['budget']}: {changed}"
        )
    if sum(recorder.visits.values()) != metrics["frame_count"]:
        raise ValueError("queue visit counter does not reconstruct the source frame count")
    if not math.isclose(
        sum(recorder.duration_by_queue.values()), metrics["elapsed_including_drain_s"],
        rel_tol=0.0, abs_tol=1e-8,
    ):
        raise ValueError("queue duration counter does not reconstruct source elapsed time")
    record = {
        "load": source_row["load"], "seed": source_row["seed"],
        "source_budget": source_row["budget"], "source_reproduced": True,
        "job_count": metrics["job_count"], "frame_count": metrics["frame_count"],
        "elapsed_including_drain_s": metrics["elapsed_including_drain_s"],
        "exact_selection_frame_count": metrics["exact_selection_frame_count"],
        "candidate_attempts": metrics["candidate_attempts"],
        "bound_option_probes": metrics["bound_option_probes"],
        "bound_threshold_evaluations": metrics["bound_threshold_evaluations"],
        "local_score_evaluations": metrics["local_score_evaluations"],
        "feasible_candidate_evaluations": metrics["feasible_candidate_evaluations"],
        "peak_search_frontier": metrics["peak_search_frontier"],
        "maximum_candidate_attempts_per_frame": metrics["maximum_candidate_attempts_per_frame"],
        "unique_queue_state_count": len(recorder.visits),
        "compared_metric_field_count": len(source_row) - 2,
    }
    return record, recorder.visits, recorder.duration_by_queue


def _physical_signature(model, selection):
    offsets = tuple(
        tuple(sorted(
            offset
            for option_index in selection.option_indices if option_index is not None
            for offset in model.options[option_index].completion_offsets[coordinate]
        ))
        for coordinate in range(len(model.workloads))
    )
    return selection.service, selection.duration, offsets


def _selection_record(selection):
    return asdict(selection)


def _trace_state(model, queue, population_visits):
    traced = trace_budgeted_frame_search(
        model.options, queue, model.lanes,
        small_budget=SMALL_BUDGET, exact_budget=EXACT_BUDGET,
    )
    small_shadow = select_budgeted_frame(model.options, queue, model.lanes, SMALL_BUDGET)
    exact_shadow = select_budgeted_frame(model.options, queue, model.lanes, EXACT_BUDGET)
    small_matches = traced.small == small_shadow
    exact_matches = traced.exact == exact_shadow
    if not (small_matches and exact_matches):
        raise ValueError(f"single trace does not reproduce legacy selectors at queue {queue}")
    regret = traced.exact.score - traced.small.score
    slack = traced.small.upper_bound - traced.exact.score
    identity_error = traced.small.gap - regret - slack
    if (not traced.exact.exact or regret < -SCORE_TOLERANCE
            or slack < -SCORE_TOLERANCE or abs(identity_error) > SCORE_TOLERANCE):
        raise ValueError(f"support-gap decomposition failed at queue {queue}")
    score_match = traced.small.score == traced.exact.score
    if (traced.first_exact_score_attempt <= SMALL_BUDGET) != score_match:
        raise ValueError(f"exact-score discovery rank is inconsistent at queue {queue}")
    physical_match = _physical_signature(model, traced.small) == _physical_signature(model, traced.exact)
    return {
        "queue": queue, "queue_norm": sum(queue),
        "population_visits": population_visits,
        "total_frame_visits": sum(row["frame_count"] for row in population_visits),
        "total_visited_time_s": sum(row["visited_time_s"] for row in population_visits),
        "k26": _selection_record(traced.small), "exact": _selection_record(traced.exact),
        "finite_budget_support_regret": max(0.0, regret),
        "bound_relaxation_slack": max(0.0, slack),
        "k26_gap": traced.small.gap,
        "certified_gap_identity_error": identity_error,
        "first_exact_score_attempt": traced.first_exact_score_attempt,
        "first_k120_selected_physical_signature_attempt": (
            traced.first_k120_selected_physical_signature_attempt
        ),
        "exact_score_found_by_k26": score_match,
        "k120_selected_physical_signature_found_by_k26": (
            traced.first_k120_selected_physical_signature_attempt <= SMALL_BUDGET
        ),
        "k26_exact_certified": traced.small.exact,
        "k26_exact_physical_action": physical_match,
        "trace_matches_separate_k26_selector": small_matches,
        "trace_matches_separate_exact_selector": exact_matches,
        "separate_validation_candidate_attempts": small_shadow.attempts + exact_shadow.attempts,
        "separate_validation_bound_option_probes": (
            small_shadow.bound_option_probes + exact_shadow.bound_option_probes
        ),
        "separate_validation_bound_threshold_evaluations": (
            small_shadow.bound_threshold_evaluations
            + exact_shadow.bound_threshold_evaluations
        ),
        "trace_physical_signature_evaluations": traced.physical_signature_evaluations,
        "trace_physical_signature_completion_offset_reads": (
            traced.physical_signature_completion_offset_reads
        ),
        "gap_class": (
            "strict_support_loss" if not score_match else
            "exact_score_found_bound_open" if not traced.small.exact else
            "exact_certified_by_k26"
        ),
    }


def _weighted_quantile(rows, field, quantile):
    ordered = sorted((row[field], row["frame_count"]) for row in rows)
    target = quantile * sum(count for _, count in ordered)
    cumulative = 0
    for value, count in ordered:
        cumulative += count
        if cumulative >= target:
            return value
    return ordered[-1][0]


def _population_summary(load, source_budget, counter, durations, trace_by_queue):
    rows = []
    for queue, count in counter.items():
        state = trace_by_queue[queue]
        rows.append({
            "state": state, "frame_count": count, "visited_time_s": durations[queue],
            "first_exact_score_attempt": state["first_exact_score_attempt"],
        })
    frame_count = sum(row["frame_count"] for row in rows)
    visited_time = sum(row["visited_time_s"] for row in rows)

    def frames_where(predicate):
        return sum(row["frame_count"] for row in rows if predicate(row["state"]))

    def time_where(predicate):
        return sum(row["visited_time_s"] for row in rows if predicate(row["state"]))

    def weighted(field, *, per_job=False):
        total = 0.0
        for row in rows:
            state = row["state"]
            value = state[field]
            if per_job and state["queue_norm"]:
                value /= state["queue_norm"]
            total += row["frame_count"] * value
        return total / frame_count

    strict = lambda state: state["gap_class"] == "strict_support_loss"
    bound_only = lambda state: state["gap_class"] == "exact_score_found_bound_open"
    top = sorted(
        (row for row in rows if strict(row["state"])),
        key=lambda row: (-row["frame_count"], row["state"]["queue"]),
    )[:8]
    return {
        "load": load, "source_budget": source_budget,
        "population_role": (
            "primary frozen K26 policy state distribution" if source_budget == SMALL_BUDGET
            else "frozen exact-policy state-distribution sensitivity"
        ),
        "frame_count": frame_count, "visited_time_s": visited_time,
        "unique_queue_state_count": len(rows),
        "strict_support_loss_frame_count": frames_where(strict),
        "strict_support_loss_frame_fraction": frames_where(strict) / frame_count,
        "strict_support_loss_time_fraction": time_where(strict) / visited_time,
        "exact_score_found_by_k26_frame_count": frames_where(
            lambda state: state["exact_score_found_by_k26"]
        ),
        "exact_score_found_by_k26_frame_fraction": frames_where(
            lambda state: state["exact_score_found_by_k26"]
        ) / frame_count,
        "exact_score_found_bound_open_frame_count": frames_where(bound_only),
        "exact_score_found_bound_open_frame_fraction": frames_where(bound_only) / frame_count,
        "k26_exact_certified_frame_count": frames_where(
            lambda state: state["k26_exact_certified"]
        ),
        "k26_exact_certified_frame_fraction": frames_where(lambda state: state["k26_exact_certified"]) / frame_count,
        "exact_physical_action_frame_fraction": frames_where(lambda state: state["k26_exact_physical_action"]) / frame_count,
        "exact_score_first_found_after_k26_frame_fraction": frames_where(lambda state: state["first_exact_score_attempt"] > SMALL_BUDGET) / frame_count,
        "k120_selected_physical_signature_first_found_after_k26_frame_fraction": frames_where(lambda state: state["first_k120_selected_physical_signature_attempt"] > SMALL_BUDGET) / frame_count,
        "mean_finite_budget_support_regret": weighted("finite_budget_support_regret"),
        "mean_finite_budget_support_regret_per_visible_job": weighted("finite_budget_support_regret", per_job=True),
        "mean_bound_relaxation_slack": weighted("bound_relaxation_slack"),
        "mean_bound_relaxation_slack_per_visible_job": weighted("bound_relaxation_slack", per_job=True),
        "mean_certified_gap": weighted("k26_gap"),
        "median_first_exact_score_attempt": _weighted_quantile(rows, "first_exact_score_attempt", .5),
        "p90_first_exact_score_attempt": _weighted_quantile(rows, "first_exact_score_attempt", .9),
        "maximum_first_exact_score_attempt": max(row["first_exact_score_attempt"] for row in rows),
        "most_frequent_strict_loss_states": [
            {
                "queue": row["state"]["queue"], "frame_count": row["frame_count"],
                "visited_time_s": row["visited_time_s"],
                "first_exact_score_attempt": row["state"]["first_exact_score_attempt"],
                "finite_budget_support_regret": row["state"]["finite_budget_support_regret"],
                "bound_relaxation_slack": row["state"]["bound_relaxation_slack"],
                "k26_option_indices": row["state"]["k26"]["option_indices"],
                "exact_option_indices": row["state"]["exact"]["option_indices"],
            }
            for row in top
        ],
    }


def _execution_checks(source_runs, state_rows, summaries, sources):
    return {
        "all_18_frozen_source_trajectories_reproduced": (
            len(source_runs) == 18 and all(row["source_reproduced"] for row in source_runs)
        ),
        "registered_source_roster_executed_once": (
            sorted((row["load"], row["seed"], row["source_budget"]) for row in source_runs)
            == [(load, seed, budget) for load in LOADS for seed in SEEDS
                for budget in (SMALL_BUDGET, EXACT_BUDGET)]
        ),
        "source_queue_counts_reconstruct_every_frame": all(
            row["frame_count"] > 0 and row["unique_queue_state_count"] > 0
            for row in source_runs
        ),
        "per_trajectory_state_records_reconstruct_frames_and_time": all(
            row["frame_count"] == sum(
                visit["frame_count"]
                for state in state_rows for visit in state["population_visits"]
                if visit["load"] == row["load"]
                and visit["seed"] == row["seed"]
                and visit["source_budget"] == row["source_budget"]
            )
            and math.isclose(
                row["elapsed_including_drain_s"],
                sum(
                    visit["visited_time_s"]
                    for state in state_rows for visit in state["population_visits"]
                    if visit["load"] == row["load"]
                    and visit["seed"] == row["seed"]
                    and visit["source_budget"] == row["source_budget"]
                ),
                rel_tol=0.0, abs_tol=1e-8,
            )
            for row in source_runs
        ),
        "source_exact_reference_remains_exact": all(
            row["exact_selection_frame_count"] == row["frame_count"]
            for row in source_runs if row["source_budget"] == EXACT_BUDGET
        ),
        "single_trace_reproduces_both_legacy_budgets": all(
            row["trace_matches_separate_k26_selector"]
            and row["trace_matches_separate_exact_selector"] for row in state_rows
        ),
        "trace_exact_reference_closes_within_k120": all(
            row["exact"]["exact"] and row["exact"]["attempts"] <= EXACT_BUDGET
            for row in state_rows
        ),
        "support_bound_and_gap_identity_hold": all(
            row["k26"]["score"] <= row["exact"]["score"] + SCORE_TOLERANCE
            and row["exact"]["score"] <= row["k26"]["upper_bound"] + SCORE_TOLERANCE
            and abs(row["certified_gap_identity_error"]) <= SCORE_TOLERANCE
            for row in state_rows
        ),
        "exact_score_discovery_rank_matches_k26_support": all(
            (row["first_exact_score_attempt"] <= SMALL_BUDGET)
            == row["exact_score_found_by_k26"] for row in state_rows
        ),
        "all_unique_visited_states_traced_once": (
            len(state_rows) == len({tuple(row["queue"]) for row in state_rows})
            and all(row["total_frame_visits"] > 0 for row in state_rows)
        ),
        "primary_and_sensitivity_populations_remain_separate": (
            len(summaries) == 6
            and [(row["load"], row["source_budget"]) for row in summaries]
            == [(load, budget) for load in LOADS for budget in (SMALL_BUDGET, EXACT_BUDGET)]
        ),
        "population_summaries_reconstruct_source_frames_and_time": all(
            summary["frame_count"] == sum(
                row["frame_count"] for row in source_runs
                if row["load"] == summary["load"]
                and row["source_budget"] == summary["source_budget"]
            )
            and math.isclose(
                summary["visited_time_s"],
                sum(
                    row["elapsed_including_drain_s"] for row in source_runs
                    if row["load"] == summary["load"]
                    and row["source_budget"] == summary["source_budget"]
                ),
                rel_tol=0.0, abs_tol=1e-7,
            )
            for summary in summaries
        ),
        "primary_k26_certified_counts_reproduce_source_metrics": all(
            summary["k26_exact_certified_frame_count"] == sum(
                row["exact_selection_frame_count"] for row in source_runs
                if row["load"] == summary["load"]
                and row["source_budget"] == SMALL_BUDGET
            )
            for summary in summaries if summary["source_budget"] == SMALL_BUDGET
        ),
        "offline_trace_adds_no_randomness": True,
        "registered_sources_unchanged": all((ROOT / name).read_text() == text for name, text in sources.items()),
    }


def register(output=DEFAULT_OUTPUT):
    output = Path(output)
    protocol_path = output.with_name(output.stem + "_protocol.json")
    snapshot_path = output.with_name(output.stem + "_fixed_sources.json")
    if any(path.exists() for path in (output, protocol_path, snapshot_path)):
        raise ValueError("candidate-search diagnostic artifacts already exist")
    development = json.loads(DEVELOPMENT.read_text())
    model, certificates, rows = _validate_development(development)
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "development search-mechanism diagnosis on consumed v4 trajectories only",
        "source_experiment": DEVELOPMENT.name,
        "loads": LOADS, "seeds": SEEDS, "arrival_horizon_s": HORIZON,
        "source_budgets": (SMALL_BUDGET, EXACT_BUDGET),
        "small_budget": SMALL_BUDGET, "exact_budget": EXACT_BUDGET,
        "model": model.snapshot(), "certificates": certificates,
        "source_rows": rows,
        "state_roster": "all queues visited by each of the 18 frozen K26/K120 source trajectories; deduplicate only for offline trace execution",
        "primary_population": "K26 trajectory queues, aggregated separately by load",
        "sensitivity_population": "K120 trajectory queues, aggregated separately by load and never pooled into the primary result",
        "trace": "one deterministic legacy v4 K120 trace per unique queue; recover K26 from the same prefix and compare both against separate frozen selector calls",
        "gap_decomposition": "U26-S26 = (Sstar-S26) finite-budget support regret + (U26-Sstar) frontier-bound slack; neither term alone identifies priority-order versus bound-pruning causality",
        "paid_attempt_rank": "idle, every homogeneous seed including infeasible seeds, and every popped complete-candidate feasibility attempt use the frozen v4 accounting",
        "score_rank": "first_exact_score_attempt is the first feasible candidate with exactly the final K120 support score under the same arithmetic",
        "physical_signature_rank": "first_k120_selected_physical_signature_attempt is the first feasible occurrence of the final K120-selected service, duration and completion-offset signature; it is not an incumbent time or a unique physical optimum",
        "stopping_rule": "run the fixed roster once; choose no new policy, thresholds, seeds, confirmation data or paper change in this diagnostic",
    }
    sources = _source_texts()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_new(protocol_path, protocol)
    _write_new(snapshot_path, sources)
    return protocol


def run_registered(output=DEFAULT_OUTPUT):
    output = Path(output)
    protocol_path = output.with_name(output.stem + "_protocol.json")
    snapshot_path = output.with_name(output.stem + "_fixed_sources.json")
    if output.exists():
        raise ValueError("candidate-search diagnostic result already exists")
    protocol = json.loads(protocol_path.read_text())
    sources = json.loads(snapshot_path.read_text())
    expected = {
        "loads": list(LOADS), "seeds": list(SEEDS),
        "arrival_horizon_s": HORIZON,
        "source_budgets": [SMALL_BUDGET, EXACT_BUDGET],
        "small_budget": SMALL_BUDGET, "exact_budget": EXACT_BUDGET,
    }
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("registered candidate-search roster or budgets changed")
    if any((ROOT / name).read_text() != text for name, text in sources.items()):
        raise ValueError("registered candidate-search sources changed")
    development = json.loads(DEVELOPMENT.read_text())
    model, certificates, rows = _validate_development(development)
    if (_json(model.snapshot()) != protocol["model"]
            or _json(certificates) != protocol["certificates"]
            or rows != protocol["source_rows"]):
        raise ValueError("registered candidate-search model, certificates or source rows changed")

    source_by_id = {
        (row["load"], row["seed"], row["budget"]): row for row in rows
    }
    source_runs = []
    counters = defaultdict(Counter)
    durations = defaultdict(Counter)
    trajectory_counters = {}
    trajectory_durations = {}
    for load, certificate in zip(LOADS, certificates, strict=True):
        for seed in SEEDS:
            tape = poisson_arrivals(certificate["arrival_rates"], HORIZON, seed)
            for budget in (SMALL_BUDGET, EXACT_BUDGET):
                source = source_by_id[(load, seed, budget)]
                record, visits, time_by_queue = _replay_source(
                    model, certificate, source, tape,
                )
                source_runs.append(record)
                counters[(load, budget)].update(visits)
                durations[(load, budget)].update(time_by_queue)
                trajectory_counters[(load, seed, budget)] = visits
                trajectory_durations[(load, seed, budget)] = time_by_queue
            print(json.dumps({"load": load, "seed": seed, "source_trajectories_reproduced": 2}), flush=True)

    all_queues = sorted(set().union(*(set(counter) for counter in counters.values())))
    state_rows = []
    for queue in all_queues:
        visits = [
            {
                "load": load, "seed": seed, "source_budget": budget,
                "frame_count": trajectory_counters[(load, seed, budget)][queue],
                "visited_time_s": trajectory_durations[(load, seed, budget)][queue],
            }
            for load in LOADS for seed in SEEDS
            for budget in (SMALL_BUDGET, EXACT_BUDGET)
            if trajectory_counters[(load, seed, budget)][queue]
        ]
        state_rows.append(_trace_state(model, queue, visits))
    trace_by_queue = {tuple(row["queue"]): row for row in state_rows}
    summaries = [
        _population_summary(load, budget, counters[(load, budget)],
                            durations[(load, budget)], trace_by_queue)
        for load in LOADS for budget in (SMALL_BUDGET, EXACT_BUDGET)
    ]
    checks = _execution_checks(source_runs, state_rows, summaries, sources)
    report = {
        "experiment": "aligned_hardware_candidate_search_diagnostic_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scientific_role": "mechanism diagnosis only; no policy advance decision",
        "checks": checks, "protocol": protocol,
        "source_trajectory_replays": source_runs,
        "summary_by_load_and_source_policy": summaries,
        "state_diagnostics": state_rows,
        "work_accounting": {
            "source_trajectory_replay": {
                "arrival_tapes_generated": len(LOADS) * len(SEEDS),
                "trajectory_count": len(source_runs),
                "selector_calls": sum(row["frame_count"] for row in source_runs),
                "candidate_attempts": sum(row["candidate_attempts"] for row in source_runs),
                "local_score_evaluations": sum(row["local_score_evaluations"] for row in source_runs),
                "bound_option_probes": sum(row["bound_option_probes"] for row in source_runs),
                "bound_threshold_evaluations": sum(row["bound_threshold_evaluations"] for row in source_runs),
                "feasible_candidate_evaluations": sum(row["feasible_candidate_evaluations"] for row in source_runs),
                "peak_search_frontier": max(row["peak_search_frontier"] for row in source_runs),
            },
            "single_continuous_k120_trace": {
                "unique_queue_states": len(state_rows),
                "selector_trace_calls": len(state_rows),
                "candidate_attempts": sum(row["exact"]["attempts"] for row in state_rows),
                "local_score_evaluations": sum(row["exact"]["local_score_evaluations"] for row in state_rows),
                "bound_option_probes": sum(row["exact"]["bound_option_probes"] for row in state_rows),
                "bound_threshold_evaluations": sum(row["exact"]["bound_threshold_evaluations"] for row in state_rows),
                "feasible_candidate_evaluations": sum(row["exact"]["feasible_candidate_count"] for row in state_rows),
                "peak_search_frontier": max(row["exact"]["frontier_peak"] for row in state_rows),
                "physical_signature_evaluations": sum(row["trace_physical_signature_evaluations"] for row in state_rows),
                "physical_signature_completion_offset_reads": sum(
                    row["trace_physical_signature_completion_offset_reads"]
                    for row in state_rows
                ),
            },
            "separate_k26_k120_selector_validation": {
                "selector_calls": 2 * len(state_rows),
                "candidate_attempts": sum(row["separate_validation_candidate_attempts"] for row in state_rows),
                "local_score_evaluations": sum(
                    row["k26"]["local_score_evaluations"]
                    + row["exact"]["local_score_evaluations"] for row in state_rows
                ),
                "bound_option_probes": sum(row["separate_validation_bound_option_probes"] for row in state_rows),
                "bound_threshold_evaluations": sum(row["separate_validation_bound_threshold_evaluations"] for row in state_rows),
                "feasible_candidate_evaluations": sum(
                    row["k26"]["feasible_candidate_count"]
                    + row["exact"]["feasible_candidate_count"] for row in state_rows
                ),
                "peak_search_frontier": max(
                    max(row["k26"]["frontier_peak"], row["exact"]["frontier_peak"])
                    for row in state_rows
                ),
            },
            "runner_selected_action_physical_comparison": {
                "signature_evaluations": 2 * len(state_rows),
                "completion_offset_reads": sum(
                    sum(
                        len(model.options[index].completion_offsets[coordinate])
                        for index in row[key]["option_indices"] if index is not None
                        for coordinate in range(len(model.workloads))
                    )
                    for row in state_rows for key in ("k26", "exact")
                ),
            },
            "offline_trace_extra_rng_initializations": 0,
            "offline_trace_extra_random_draws": 0,
        },
        "development_advance": False,
        "limitations": [
            "K120 certifies maximum one-frame support for this finite catalogue; it is not a mean-flow oracle.",
            "Queue frequencies are policy-induced. K26 is the primary development distribution and K120 is a separate sensitivity population.",
            "The winner rank records the realized legacy search path on visited states. Because bounds also determine heap expansion, it does not isolate priority-order causality; frontier slack does not identify one responsible option or subtree.",
            "All trajectories reuse development seeds 41--43. This diagnostic supplies no independent confirmation and changes no scheduler or manuscript claim.",
        ],
    }
    if not all((ROOT / name).read_text() == text for name, text in sources.items()):
        raise ValueError("registered candidate-search sources changed during execution")
    _write_new(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = register(args.output) if args.register else run_registered(args.output)
    print(json.dumps({
        "status": "REGISTERED" if args.register else result["status"],
        "output": str(args.output),
    }))
    raise SystemExit(0 if args.register or result["status"] == "PASS" else 1)
