"""Registered full development trajectories for the frozen residual sidecar.

The v4 K26/K120 rows and their flow decompositions are reused as immutable
controls.  The only new stochastic executions are the residual-sidecar K26
policy on the same three loads and three already-used arrival seeds.
Registration never constructs an arrival tape or calls the selector.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from algorithm.residual_sidecar_frame_candidates import select_residual_sidecar_frame
from algorithm.experiments.aligned_frame_flow_decomposition import decompose_flow
from algorithm.experiments.aligned_hardware_frame_confirmation import _write_new
from algorithm.experiments.aligned_hardware_frame_experiment import (
    poisson_arrivals,
    simulate_frames,
)
from algorithm.experiments.aligned_hardware_frame_model import (
    capacity_certificate,
    load_model,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACTS / (
    "aligned_hardware_residual_sidecar_trajectories_development_20260911.json"
)
BASELINE_RESULT = ARTIFACTS / "aligned_hardware_frame_experiment_v4_20260909.json"
BASELINE_MECHANISM = ARTIFACTS / (
    "aligned_hardware_frame_mechanism_development_20260909.json"
)
SOURCE_DIAGNOSTIC = ARTIFACTS / (
    "aligned_hardware_candidate_search_diagnostic_development_20260910.json"
)
SOURCE_SHADOW_V1 = ARTIFACTS / (
    "aligned_hardware_residual_sidecar_shadow_development_20260911.json"
)
SOURCE_SHADOW_V2 = ARTIFACTS / (
    "aligned_hardware_residual_sidecar_shadow_v2_development_20260911.json"
)
SOURCE_SHADOW_V2_PROTOCOL = ARTIFACTS / (
    "aligned_hardware_residual_sidecar_shadow_v2_development_20260911_protocol.json"
)
SOURCE_SHADOW_V2_FIXED_SOURCES = ARTIFACTS / (
    "aligned_hardware_residual_sidecar_shadow_v2_development_20260911_fixed_sources.json"
)

LOADS = (0.50, 0.80, 0.95)
SEEDS = (41, 42, 43)
SMALL_BUDGET = 26
EXACT_BUDGET = 120
BASELINE_BUDGETS = (SMALL_BUDGET, EXACT_BUDGET)
HORIZON = 20_000.0
EXECUTION_WORKERS = 3
GATE_TOLERANCE = 1e-12

DECOMPOSITION_FIELDS = (
    "mean_deferred_visible_wait_s",
    "mean_arrival_barrier_wait_s",
    "mean_dispatch_to_completion_s",
)
SIDECAR_SUM_FIELDS = (
    "attempts",
    "seed_attempts",
    "feasible_candidate_count",
    "local_score_evaluations",
    "bound_option_probes",
    "bound_threshold_evaluations",
    "upper_bound_calls",
    "bound_price_vector_count",
    "bound_dual_table_option_probes",
    "bound_dual_mask_evaluations",
    "bound_dual_mask_sums",
    "bound_dual_coordinate_products",
    "bound_strict_tightening_count",
    "bound_tightening_sum",
    "safe_pruned_prefixes_on_generation",
    "safe_pruned_prefixes_on_pop",
    "legacy_rejected_prefixes",
    "frontier_pushes",
    "paid_search_pops",
    "exact_stop_frontier_entries_cleared",
    "virtual_legacy_serial_allocations",
    "certificate_frontier_scans",
    "certificate_frontier_entries",
)
NEW_SOURCES = (
    "algorithm/experiments/aligned_frame_flow_decomposition.py",
    "algorithm/experiments/aligned_hardware_frame_mechanism.py",
    "algorithm/experiments/aligned_hardware_residual_sidecar_trajectories.py",
)


def _json(value):
    return json.loads(json.dumps(value))


def _paths(output):
    output = Path(output)
    return (
        output,
        output.with_name(output.stem + "_protocol.json"),
        output.with_name(output.stem + "_fixed_sources.json"),
    )


def _compact_decomposition(decomposition):
    return {
        "mean_flow_s": decomposition["mean_flow_s"],
        **{field: decomposition[field] for field in DECOMPOSITION_FIELDS},
        "arrival_window_mean_flow_contribution_s": decomposition["time_regions"][
            "arrival_window"
        ]["mean_flow_contribution_s"],
        "drain_mean_flow_contribution_s": decomposition["time_regions"]["drain"][
            "mean_flow_contribution_s"
        ],
        "frame_duration_mean_s": decomposition["frame_duration_mean_s"],
        "frame_duration_time_weighted_mean_s": decomposition[
            "frame_duration_time_weighted_mean_s"
        ],
        "occupied_lane_seconds": decomposition["occupied_lane_seconds"],
        "global_barrier_idle_lane_seconds": decomposition[
            "global_barrier_idle_lane_seconds"
        ],
        "by_class": {
            name: {
                "mean_flow_s": row["mean_flow_s"],
                **{field: row[field] for field in DECOMPOSITION_FIELDS},
            }
            for name, row in decomposition["by_class"].items()
        },
    }


def _baseline_context():
    baseline = json.loads(BASELINE_RESULT.read_text())
    if (
        baseline.get("experiment") != "aligned_hardware_frame_and_budget_v4"
        or baseline.get("status") != "PASS"
        or not all(baseline.get("checks", {}).values())
        or baseline.get("evaluation_role")
        != "development comparison on reused seeds; independent confirmation pending"
        or baseline["protocol"]["arrival_horizon_s"] != HORIZON
        or tuple(baseline["protocol"]["loads"]) != LOADS
        or tuple(baseline["protocol"]["seeds"]) != SEEDS
        or tuple(baseline["protocol"]["budgets"]) != (10, 14, 26, 120)
        or baseline["protocol"]["exact_reference_budget"] != EXACT_BUDGET
        or baseline["protocol"]["dispatch"]
        != "global synchronous frames; released jobs only; FIFO within class"
    ):
        raise ValueError("full sidecar trajectories require the frozen v4 development result")

    baseline_rows = [
        row for row in baseline["runs"] if row["budget"] in BASELINE_BUDGETS
    ]
    expected = {
        (load, seed, budget)
        for load in LOADS
        for seed in SEEDS
        for budget in BASELINE_BUDGETS
    }
    if (
        len(baseline_rows) != len(expected)
        or {(r["load"], r["seed"], r["budget"]) for r in baseline_rows} != expected
    ):
        raise ValueError("frozen v4 K26/K120 trajectory roster changed")
    baseline_rows.sort(key=lambda row: (row["load"], row["seed"], row["budget"]))

    diagnostic = json.loads(SOURCE_DIAGNOSTIC.read_text())
    replay_keys = {
        (row["load"], row["seed"], row["source_budget"])
        for row in diagnostic.get("source_trajectory_replays", [])
        if row.get("source_reproduced")
    }
    if (
        diagnostic.get("experiment")
        != "aligned_hardware_candidate_search_diagnostic_v1"
        or diagnostic.get("status") != "PASS"
        or not all(diagnostic.get("checks", {}).values())
        or replay_keys != expected
        or len(diagnostic["source_trajectory_replays"]) != len(expected)
    ):
        raise ValueError("the 18 frozen v4 trajectory replays are not intact")

    mechanism = json.loads(BASELINE_MECHANISM.read_text())
    if (
        mechanism.get("experiment") != "aligned_frame_flow_mechanism_development"
        or mechanism.get("status") != "PASS"
        or not all(mechanism.get("checks", {}).values())
        or tuple(mechanism["protocol"]["loads"]) != LOADS
        or tuple(mechanism["protocol"]["seeds"]) != SEEDS
        or mechanism["protocol"]["arrival_horizon_s"] != HORIZON
    ):
        raise ValueError("frozen v4 flow decompositions are not intact")
    modes = {SMALL_BUDGET: "k26", EXACT_BUDGET: "exact"}
    mechanism_rows = {
        (row["load"], row["seed"], budget): row
        for budget, mode in modes.items()
        for row in mechanism["runs"]
        if row["mode"] == mode
    }
    indexed = {(r["load"], r["seed"], r["budget"]): r for r in baseline_rows}
    if set(mechanism_rows) != expected:
        raise ValueError("frozen v4 decomposition roster changed")
    compact_decompositions = []
    for key in sorted(expected):
        source = mechanism_rows[key]
        baseline_metrics = indexed[key]
        changed = [
            name
            for name, value in baseline_metrics.items()
            if name not in ("load", "seed") and source["metrics"].get(name) != value
        ]
        if changed or abs(
            source["decomposition"]["mean_flow_s"] - baseline_metrics["mean_flow_s"]
        ) > 1e-8:
            raise ValueError(f"frozen v4 decomposition changed at {key}: {changed}")
        compact_decompositions.append(
            {
                "load": key[0],
                "seed": key[1],
                "budget": key[2],
                **_compact_decomposition(source["decomposition"]),
            }
        )
    return baseline, baseline_rows, compact_decompositions


def _shadow_context():
    v1 = json.loads(SOURCE_SHADOW_V1.read_text())
    v2 = json.loads(SOURCE_SHADOW_V2.read_text())
    v2_protocol = json.loads(SOURCE_SHADOW_V2_PROTOCOL.read_text())
    unchanged_science = (
        "strict_bound_activation_state_count",
        "safe_pruning_activation_state_count",
        "generation_pruning_state_count",
        "lazy_pop_pruning_state_count",
        "support_improved_state_count",
        "support_regressed_state_count",
        "certificate_improved_state_count",
        "candidate_attempts_reduced_state_count",
        "physical_action_changed_state_count",
        "retained_change_state_count",
        "shadow_advance",
        "certificate_or_work_only",
        "scientific_decision",
    )
    if (
        v1.get("experiment") != "residual_capacity_binary_price_sidecar_shadow_v1"
        or v2.get("experiment") != "residual_capacity_binary_price_sidecar_shadow_v2"
        or v1.get("status") != v2.get("status")
        or v2.get("status") != "PASS"
        or not all(v1.get("checks", {}).values())
        or not all(v2.get("checks", {}).values())
        or v1.get("state_evaluation_count") != 5003
        or v2.get("state_evaluation_count") != 5003
        or v2.get("supersedes_result") != SOURCE_SHADOW_V1.name
        or v2_protocol.get("supersedes_result") != SOURCE_SHADOW_V1.name
        or v2.get("correction") != v2_protocol.get("correction")
        or v2.get("scientific_decision")
        != "ADVANCE_TO_FULL_DEVELOPMENT_TRAJECTORIES"
        or not v2.get("shadow_advance")
        or v2.get("support_regressed_state_count") != 0
        or any(v1.get(field) != v2.get(field) for field in unchanged_science)
        or v1["work_accounting"]["continuous_trace"][
            "certificate_frontier_scans"
        ]
        != 5003
        or v2["work_accounting"]["continuous_trace"][
            "certificate_frontier_scans"
        ]
        != 5189
        or v2["work_accounting"]["continuous_trace"][
            "certificate_frontier_entries_scanned"
        ]
        != 569
    ):
        raise ValueError("the corrected residual-sidecar development lineage is not intact")
    return {
        "v1_result": SOURCE_SHADOW_V1.name,
        "v2_result": SOURCE_SHADOW_V2.name,
        "v2_protocol": SOURCE_SHADOW_V2_PROTOCOL.name,
        "v2_supersedes_v1": True,
        "state_count": 5003,
        "v1_recorded_certificate_scans": 5003,
        "v2_corrected_certificate_scans": 5189,
        "v2_certificate_entries_scanned": 569,
        "fixed_state_support_improvements": v2["support_improved_state_count"],
        "fixed_state_support_regressions": v2["support_regressed_state_count"],
        "fixed_state_physical_action_changes": v2[
            "physical_action_changed_state_count"
        ],
        "scientific_decision": v2["scientific_decision"],
        "correction": v2["correction"],
    }


def _protocol_fields():
    baseline, baseline_rows, decompositions = _baseline_context()
    shadow = _shadow_context()
    model = load_model()
    certificates = [capacity_certificate(model, load) for load in LOADS]
    if (
        _json(model.snapshot()) != baseline["model"]
        or _json(certificates) != baseline["certificates"]
    ):
        raise ValueError("registered trajectory model or capacity certificates changed")
    return {
        "experiment": "aligned_residual_sidecar_full_development_trajectories_v1",
        "role": (
            "post-outcome development on consumed v4 arrival seeds; not independent "
            "confirmation"
        ),
        "source_shadow": shadow,
        "baseline_result": BASELINE_RESULT.name,
        "baseline_mechanism_result": BASELINE_MECHANISM.name,
        "baseline_replay_validation": SOURCE_DIAGNOSTIC.name,
        "loads": list(LOADS),
        "seeds": list(SEEDS),
        "new_run_budget": SMALL_BUDGET,
        "frozen_baseline_budgets": list(BASELINE_BUDGETS),
        "small_budget": SMALL_BUDGET,
        "exact_budget": EXACT_BUDGET,
        "arrival_horizon_s": HORIZON,
        "arrival_model": baseline["protocol"]["arrival_model"],
        "dispatch": baseline["protocol"]["dispatch"],
        "model": baseline["model"],
        "certificates": baseline["certificates"],
        "frozen_v4_rows": baseline_rows,
        "frozen_v4_flow_decompositions": decompositions,
        "new_run_roster": [
            {"load": load, "seed": seed, "budget": budget}
            for load in LOADS
            for seed in SEEDS
            for budget in (SMALL_BUDGET,)
        ],
        "selector_rule": (
            "exactly one residual-sidecar selector call per frame; no legacy shadow, "
            "continuous checkpoint trace or queue-state cache"
        ),
        "online_information": (
            "selector receives only the current released queue and static frame catalogue"
        ),
        "work_accounting": (
            "paid candidate attempts and all residual dual-bound, pruning, frontier, "
            "certificate and selector wall time are counted"
        ),
        "primary_estimand": (
            "K26 sidecar/v4 mean-flow ratio paired by arrival seed; arithmetic mean "
            "of the three seed ratios within each load"
        ),
        "guardrail": (
            "K26 paired p90-flow ratio averaged within each load, plus every individual "
            "K26 seed-paired mean-flow ratio"
        ),
        "advance_criteria": {
            "mean_flow_nonworse_at_every_load": "ratio <= 1 + 1e-12",
            "p90_flow_nonworse_at_every_load": "ratio <= 1 + 1e-12",
            "all_nine_seed_paired_mean_flows_nonworse": "ratio <= 1 + 1e-12",
            "strict_mean_flow_improvement_at_some_load": "ratio < 1 - 1e-12",
            "online_mechanism_activated": "at least one K26 frame with safe pruning",
        },
        "exact_sensitivity": (
            "reuse the frozen v4 K120 trajectory as a descriptive exact-support "
            "reference; do not execute a new sidecar K120 performance arm"
        ),
        "execution_workers": EXECUTION_WORKERS,
        "stopping_rule": (
            "run all 9 registered K26 sidecar trajectories once; no budget, mask, threshold, "
            "seed or policy tuning and no confirmation data"
        ),
    }


def _source_texts():
    sources = json.loads(SOURCE_SHADOW_V2_FIXED_SOURCES.read_text())
    changed = [name for name, text in sources.items() if (ROOT / name).read_text() != text]
    if changed:
        raise ValueError(f"registered residual-sidecar v2 sources changed: {changed}")
    return {**sources, **{name: (ROOT / name).read_text() for name in NEW_SOURCES}}


def register(output=DEFAULT_OUTPUT):
    result_path, protocol_path, sources_path = _paths(output)
    if any(path.exists() for path in (result_path, protocol_path, sources_path)):
        raise ValueError("residual-sidecar trajectory artifacts already exist")
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        **_protocol_fields(),
    }
    sources = _source_texts()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    _write_new(protocol_path, protocol)
    _write_new(sources_path, sources)
    return protocol


def _work_identity_holds(selected, options, queues, lanes):
    dimensions = len(queues)
    masks = 1 << dimensions
    positions = len(options) + 1
    thresholds = len({1.0, *(max(1.0, option.duration) for option in options)})
    return (
        selected.bound_price_vector_count == masks
        and selected.bound_option_probes == thresholds * positions
        and selected.bound_dual_table_option_probes
        == thresholds * masks * positions
        and selected.bound_dual_mask_evaluations
        == selected.bound_threshold_evaluations * masks
        and selected.bound_dual_mask_sums == selected.upper_bound_calls * masks
        and selected.bound_dual_coordinate_products
        == dimensions * (masks * positions + selected.upper_bound_calls)
        and selected.upper_bound_calls
        == selected.legacy_rejected_prefixes
        + selected.virtual_legacy_serial_allocations
        and selected.virtual_legacy_serial_allocations
        == selected.safe_pruned_prefixes_on_generation + selected.frontier_pushes
        and selected.frontier_pushes
        == selected.paid_search_pops
        + selected.safe_pruned_prefixes_on_pop
        + selected.exact_stop_frontier_entries_cleared
        + selected.certificate_frontier_entries
        and selected.attempts == selected.seed_attempts + selected.paid_search_pops
        and selected.seed_attempts == positions
        and selected.certificate_frontier_scans == 1
        and len(selected.option_indices) == lanes
    )


class SidecarRecorder:
    """Select online once per frame and record the sidecar's complete work."""

    def __init__(self, budget):
        self.budget = budget
        self.now = 0.0
        self.frames = []
        self.sidecar_wall_s = 0.0
        self.sidecar_totals = dict.fromkeys(SIDECAR_SUM_FIELDS, 0)
        self.sidecar_exact_frames = 0
        self.strict_tightening_frames = 0
        self.safe_pruning_frames = 0
        self.work_identity_failures = 0
        self.maximum_tightening = 0.0
        self.peak_frontier = 0
        self.maximum_attempts = 0
        self.maximum_gap = 0.0

    def __call__(self, options, queues, lanes, budget):
        if budget != self.budget:
            raise ValueError("recorder budget changed within one trajectory")
        started = perf_counter()
        selected = select_residual_sidecar_frame(options, queues, lanes, budget)
        self.sidecar_wall_s += perf_counter() - started
        for field in SIDECAR_SUM_FIELDS:
            self.sidecar_totals[field] += getattr(selected, field)
        self.sidecar_exact_frames += int(selected.exact)
        self.strict_tightening_frames += int(selected.bound_strict_tightening_count > 0)
        safe_prunes = (
            selected.safe_pruned_prefixes_on_generation
            + selected.safe_pruned_prefixes_on_pop
        )
        self.safe_pruning_frames += int(safe_prunes > 0)
        self.maximum_tightening = max(
            self.maximum_tightening, selected.bound_tightening_max
        )
        self.peak_frontier = max(self.peak_frontier, selected.frontier_peak)
        self.maximum_attempts = max(self.maximum_attempts, selected.attempts)
        self.maximum_gap = max(self.maximum_gap, selected.gap)
        self.work_identity_failures += int(
            not _work_identity_holds(selected, options, queues, lanes)
        )

        self.frames.append(
            {
                "start_s": self.now,
                "duration_s": selected.duration,
                "queue_before": tuple(queues),
                "departures": selected.service,
                "option_indices": selected.option_indices,
            }
        )
        self.now += selected.duration
        return selected

    def accounting(self):
        count = len(self.frames)
        return {
            "frame_count": count,
            "sidecar_selector_wall_s": self.sidecar_wall_s,
            "sidecar_selector_mean_wall_ms": 1000.0 * self.sidecar_wall_s / count,
            "sidecar": dict(self.sidecar_totals),
            "sidecar_exact_frame_count": self.sidecar_exact_frames,
            "strict_tightening_frame_count": self.strict_tightening_frames,
            "safe_pruning_frame_count": self.safe_pruning_frames,
            "work_identity_failure_count": self.work_identity_failures,
            "maximum_bound_tightening": self.maximum_tightening,
            "peak_search_frontier": self.peak_frontier,
            "maximum_candidate_attempts_per_frame": self.maximum_attempts,
            "maximum_certified_candidate_gap": self.maximum_gap,
        }


def _run_trajectory(task):
    model, certificate, load, seed, tape, horizon = task
    recorder = SidecarRecorder(SMALL_BUDGET)
    metrics = simulate_frames(
        model,
        tape,
        horizon,
        certificate,
        SMALL_BUDGET,
        selector=recorder,
    )
    decomposition = decompose_flow(model, tape, recorder.frames, horizon)
    return {
        "load": load,
        "seed": seed,
        "budget": SMALL_BUDGET,
        "metrics": metrics,
        "decomposition": _compact_decomposition(decomposition),
        "selector_accounting": recorder.accounting(),
    }


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize_runs(runs, protocol):
    baseline = {
        (row["load"], row["seed"], row["budget"]): row
        for row in protocol["frozen_v4_rows"]
    }
    baseline_decomposition = {
        (row["load"], row["seed"], row["budget"]): row
        for row in protocol["frozen_v4_flow_decompositions"]
    }
    paired = []
    for row in runs:
        key = (row["load"], row["seed"], row["budget"])
        old = baseline[key]
        old_decomposition = baseline_decomposition[key]
        exact_key = (row["load"], row["seed"], EXACT_BUDGET)
        exact = baseline[exact_key]
        exact_decomposition = baseline_decomposition[exact_key]
        metrics = row["metrics"]
        ratios = {
            field: _ratio(metrics[field], old[field])
            for field in (
                "mean_flow_s",
                "p90_flow_s",
                "makespan_s",
                "time_average_unfinished_jobs",
                "frame_boundary_mean_backlog",
                "candidate_attempts",
            )
        }
        component_deltas = {
            field: row["decomposition"][field] - old_decomposition[field]
            for field in DECOMPOSITION_FIELDS
        }
        exact_component_deltas = {
            field: row["decomposition"][field] - exact_decomposition[field]
            for field in DECOMPOSITION_FIELDS
        }
        paired.append(
            {
                "load": row["load"],
                "seed": row["seed"],
                "budget": row["budget"],
                "job_count": metrics["job_count"],
                "sidecar_over_v4": ratios,
                "sidecar_over_v4_exact": {
                    field: _ratio(metrics[field], exact[field])
                    for field in ("mean_flow_s", "p90_flow_s", "makespan_s")
                },
                "mean_flow_delta_s": metrics["mean_flow_s"] - old["mean_flow_s"],
                "p90_flow_delta_s": metrics["p90_flow_s"] - old["p90_flow_s"],
                "component_deltas_s": component_deltas,
                "component_delta_identity_error_s": sum(component_deltas.values())
                - (metrics["mean_flow_s"] - old["mean_flow_s"]),
                "component_deltas_over_v4_exact_s": exact_component_deltas,
                "exact_component_delta_identity_error_s": sum(
                    exact_component_deltas.values()
                ) - (metrics["mean_flow_s"] - exact["mean_flow_s"]),
                "sidecar_safe_pruning_frames": row["selector_accounting"][
                    "safe_pruning_frame_count"
                ],
            }
        )
    paired.sort(key=lambda row: (row["load"], row["seed"], row["budget"]))

    summaries = []
    for load in LOADS:
        group = [row for row in paired if row["load"] == load]
        flow_ratios = [row["sidecar_over_v4"]["mean_flow_s"] for row in group]
        p90_ratios = [row["sidecar_over_v4"]["p90_flow_s"] for row in group]
        summaries.append(
            {
                "load": load,
                "budget": SMALL_BUDGET,
                "seeds": [row["seed"] for row in group],
                "mean_paired_flow_ratio_over_v4": mean(flow_ratios),
                "paired_flow_ratio_range": [min(flow_ratios), max(flow_ratios)],
                "improved_seed_count": sum(
                    ratio < 1.0 - GATE_TOLERANCE for ratio in flow_ratios
                ),
                "equal_seed_count": sum(
                    abs(ratio - 1.0) <= GATE_TOLERANCE for ratio in flow_ratios
                ),
                "worse_seed_count": sum(
                    ratio > 1.0 + GATE_TOLERANCE for ratio in flow_ratios
                ),
                "mean_paired_p90_ratio_over_v4": mean(p90_ratios),
                "paired_p90_ratio_range": [min(p90_ratios), max(p90_ratios)],
                "mean_paired_makespan_ratio_over_v4": mean(
                    row["sidecar_over_v4"]["makespan_s"] for row in group
                ),
                "mean_paired_candidate_attempt_ratio_over_v4": mean(
                    row["sidecar_over_v4"]["candidate_attempts"] for row in group
                ),
                "mean_paired_flow_ratio_over_v4_exact": mean(
                    row["sidecar_over_v4_exact"]["mean_flow_s"] for row in group
                ),
                "mean_paired_p90_ratio_over_v4_exact": mean(
                    row["sidecar_over_v4_exact"]["p90_flow_s"] for row in group
                ),
                "mean_component_deltas_s": {
                    field: mean(row["component_deltas_s"][field] for row in group)
                    for field in DECOMPOSITION_FIELDS
                },
                "safe_pruning_frames": sum(
                    row["sidecar_safe_pruning_frames"] for row in group
                ),
            }
        )
    primary = [row for row in paired if row["budget"] == SMALL_BUDGET]
    overall = {
        "budget": SMALL_BUDGET,
        "paired_seed_count": len(primary),
        "mean_of_nine_paired_flow_ratios": mean(
            row["sidecar_over_v4"]["mean_flow_s"] for row in primary
        ),
        "paired_flow_ratio_range": [
            min(row["sidecar_over_v4"]["mean_flow_s"] for row in primary),
            max(row["sidecar_over_v4"]["mean_flow_s"] for row in primary),
        ],
        "safe_pruning_frames": sum(
            row["sidecar_safe_pruning_frames"] for row in primary
        ),
        "mean_component_deltas_s": {
            field: mean(row["component_deltas_s"][field] for row in primary)
            for field in DECOMPOSITION_FIELDS
        },
    }
    return paired, summaries, overall


def _scientific_checks(paired, summaries, execution_passed):
    primary_pairs = [row for row in paired if row["budget"] == SMALL_BUDGET]
    primary_summaries = [
        row for row in summaries if row["budget"] == SMALL_BUDGET
    ]
    return {
        "execution_checks_pass": execution_passed,
        "k26_mean_flow_nonworse_at_every_load": all(
            row["mean_paired_flow_ratio_over_v4"] <= 1.0 + GATE_TOLERANCE
            for row in primary_summaries
        ),
        "k26_p90_flow_nonworse_at_every_load": all(
            row["mean_paired_p90_ratio_over_v4"] <= 1.0 + GATE_TOLERANCE
            for row in primary_summaries
        ),
        "all_nine_k26_seed_pairs_mean_flow_nonworse": all(
            row["sidecar_over_v4"]["mean_flow_s"] <= 1.0 + GATE_TOLERANCE
            for row in primary_pairs
        ),
        "some_load_has_strict_k26_mean_flow_improvement": any(
            row["mean_paired_flow_ratio_over_v4"] < 1.0 - GATE_TOLERANCE
            for row in primary_summaries
        ),
        "k26_safe_pruning_observed": sum(
            row["sidecar_safe_pruning_frames"] for row in primary_pairs
        ) > 0,
    }


def _decision(execution_checks, scientific_checks):
    if not all(execution_checks.values()):
        return {
            "development_advance": False,
            "scientific_decision": "INVALID_EXECUTION",
            "next_step": "repair execution only under a newly versioned registration",
        }
    if all(scientific_checks.values()):
        return {
            "development_advance": True,
            "scientific_decision": "ADVANCE_TO_FRESH_SEED_CONFIRMATION",
            "next_step": (
                "register an unused arrival-seed confirmation roster; do not reuse "
                "the already consumed 1041-1050 v4 confirmation seeds"
            ),
        }
    if (
        scientific_checks["k26_mean_flow_nonworse_at_every_load"]
        and scientific_checks["k26_p90_flow_nonworse_at_every_load"]
        and scientific_checks["all_nine_k26_seed_pairs_mean_flow_nonworse"]
        and not scientific_checks[
            "some_load_has_strict_k26_mean_flow_improvement"
        ]
    ):
        return {
            "development_advance": False,
            "scientific_decision": "NO_FLOW_BENEFIT_STOP",
            "next_step": "close the sidecar trajectory branch without confirmation",
        }
    return {
        "development_advance": False,
        "scientific_decision": "TRAJECTORY_TRADEOFF_STOP",
        "next_step": (
            "close the sidecar trajectory branch without threshold, budget or seed tuning"
        ),
    }


def run_registered(output=DEFAULT_OUTPUT):
    result_path, protocol_path, sources_path = _paths(output)
    if result_path.exists():
        raise ValueError("registered residual-sidecar trajectory result already exists")
    protocol = json.loads(protocol_path.read_text())
    expected = _protocol_fields()
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("registered trajectory roster, controls or gates changed before arrivals")
    sources = json.loads(sources_path.read_text())
    if sources != _source_texts() or any(
        (ROOT / name).read_text() != text for name, text in sources.items()
    ):
        raise ValueError("registered residual-sidecar trajectory sources changed")

    model = load_model()
    certificates = [capacity_certificate(model, load) for load in LOADS]
    baseline = {
        (row["load"], row["seed"], row["budget"]): row
        for row in protocol["frozen_v4_rows"]
    }
    tapes = {}
    for load, certificate in zip(LOADS, certificates, strict=True):
        for seed in SEEDS:
            tape = poisson_arrivals(certificate["arrival_rates"], HORIZON, seed)
            counts = {
                baseline[(load, seed, budget)]["job_count"]
                for budget in BASELINE_BUDGETS
            }
            if len(counts) != 1 or len(tape) != next(iter(counts)):
                raise ValueError(
                    f"registered arrival population changed before selection at {(load, seed)}"
                )
            tapes[(load, seed)] = tape

    tasks = [
        (model, certificate, load, seed, tapes[(load, seed)], HORIZON)
        for load, certificate in zip(LOADS, certificates, strict=True)
        for seed in SEEDS
    ]
    started = perf_counter()
    runs = []
    with ProcessPoolExecutor(max_workers=EXECUTION_WORKERS) as pool:
        for row in pool.map(_run_trajectory, tasks):
            runs.append(row)
            print(
                json.dumps(
                    {
                        "load": row["load"],
                        "seed": row["seed"],
                        "sidecar_runs_completed": len(runs),
                    }
                ),
                flush=True,
            )
    runs.sort(key=lambda row: (row["load"], row["seed"], row["budget"]))

    expected_roster = {
        (load, seed, budget)
        for load in LOADS
        for seed in SEEDS
        for budget in (SMALL_BUDGET,)
    }
    actual_roster = {(r["load"], r["seed"], r["budget"]) for r in runs}
    common_accounting = all(
        row["metrics"][simulator_field] == row["selector_accounting"]["sidecar"][selector_field]
        for row in runs
        for simulator_field, selector_field in (
            ("candidate_attempts", "attempts"),
            ("bound_option_probes", "bound_option_probes"),
            ("bound_threshold_evaluations", "bound_threshold_evaluations"),
            ("local_score_evaluations", "local_score_evaluations"),
            ("feasible_candidate_evaluations", "feasible_candidate_count"),
        )
    ) and all(
        row["metrics"]["peak_search_frontier"]
        == row["selector_accounting"]["peak_search_frontier"]
        and row["metrics"]["maximum_candidate_attempts_per_frame"]
        == row["selector_accounting"]["maximum_candidate_attempts_per_frame"]
        and row["metrics"]["maximum_certified_candidate_gap"]
        == row["selector_accounting"]["maximum_certified_candidate_gap"]
        for row in runs
    )
    checks = {
        "complete_9_run_roster": len(runs) == 9 and actual_roster == expected_roster,
        "all_jobs_complete_on_registered_populations": all(
            row["metrics"]["job_count"]
            == row["metrics"]["completed_jobs"]
            == baseline[(row["load"], row["seed"], row["budget"])]["job_count"]
            for row in runs
        ),
        "flow_decomposition_matches_simulator": all(
            abs(row["metrics"]["mean_flow_s"] - row["decomposition"]["mean_flow_s"])
            <= 1e-8
            for row in runs
        ),
        "candidate_attempts_respect_registered_budgets": all(
            row["metrics"]["maximum_candidate_attempts_per_frame"] <= row["budget"]
            for row in runs
        ),
        "queue_recursion_and_drift_hold": all(
            row["metrics"]["maximum_queue_recursion_violation"] == 0
            and row["metrics"]["maximum_expected_drift_bound_violation"] <= 1e-8
            for row in runs
        ),
        "simulator_and_sidecar_common_accounting_agree": common_accounting,
        "all_sidecar_work_identities_hold": all(
            row["selector_accounting"]["work_identity_failure_count"] == 0
            for row in runs
        ),
        "registered_sources_unchanged": sources == _source_texts(),
    }
    paired, summaries, overall = summarize_runs(runs, protocol)
    checks["paired_flow_component_identity_holds"] = all(
        abs(row["component_delta_identity_error_s"]) <= 1e-8
        and abs(row["exact_component_delta_identity_error_s"]) <= 1e-8
        for row in paired
    )
    execution_passed = all(checks.values())
    scientific_checks = _scientific_checks(paired, summaries, execution_passed)
    decision = _decision(checks, scientific_checks)
    report = {
        "experiment": protocol["experiment"],
        "status": "PASS" if execution_passed else "FAIL",
        "checks": checks,
        "protocol": protocol,
        "runs": runs,
        "paired_comparisons": paired,
        "summary_by_load_and_budget": summaries,
        "overall_k26_summary": overall,
        "development_advance_checks": scientific_checks,
        **decision,
        "execution_wall_s": perf_counter() - started,
        "limitations": [
            "These are reused development arrival seeds, not independent confirmation.",
            "Candidate-attempt reductions do not establish lower wall-clock cost because the sidecar evaluates eight residual-price masks.",
        ],
    }
    _write_new(result_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = register(args.output) if args.register else run_registered(args.output)
    print(
        json.dumps(
            {
                "status": "REGISTERED" if args.register else result["status"],
                "scientific_decision": None
                if args.register
                else result["scientific_decision"],
                "output": str(args.output),
            }
        )
    )
    return 0 if args.register or result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
