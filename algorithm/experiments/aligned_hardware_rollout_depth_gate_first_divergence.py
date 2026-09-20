"""Diagnose the first gate/v4 action divergence on the failed development seed.

The diagnostic replays only the already consumed load-0.8, arrival-seed-41
trajectories.  For each planning seed it reproduces the native gated policy,
finds its first physical action difference from frozen v4, forces the two
actions from the identical prefix, and then sends both branches through the
same memoized v4 K26 continuation until the original population drains.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import accumulate
import json
from pathlib import Path

from algorithm.budgeted_frame_candidates import select_budgeted_frame
from algorithm.mixed_successor_rollout_frame_candidates import MixedSuccessorRolloutFrameSelection
from algorithm.experiments.aligned_hardware_frame_confirmation import _write_new
from algorithm.experiments.aligned_hardware_frame_experiment import poisson_arrivals
from algorithm.experiments.aligned_hardware_frame_mechanism import action_description, action_signature
from algorithm.experiments.aligned_hardware_frame_model import capacity_certificate, load_model
from algorithm.experiments.aligned_hardware_lookahead import terminal_service_rates
from algorithm.experiments.aligned_hardware_lookahead_continuation import (
    _checked_run,
    action_audit,
    paired_effect,
)
from algorithm.experiments.aligned_hardware_rollout_depth_gate import (
    GATE_CAPACITY,
    GATE_THRESHOLD,
    GATE_WEIGHTS,
    NEW_POLICY,
    RolloutDepthGateRecorder,
)
from algorithm.experiments.aligned_hardware_waiting_cost import ARTIFACTS, COMPONENTS
from algorithm.experiments.aligned_lookahead_continuation import DecisionPairs, PrefixReplay


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ARTIFACTS / "aligned_hardware_rollout_depth_gate_first_divergence_development_20260910.json"
DEVELOPMENT = ARTIFACTS / "aligned_hardware_rollout_depth_gate_development_20260910.json"
DEVELOPMENT_SOURCES = ARTIFACTS / "aligned_hardware_rollout_depth_gate_development_20260910_fixed_sources.json"
LOAD, REAL_SEED, BUDGET, PLANNING_SEEDS = .8, 41, 26, (0, 1, 2)
V4_POLICY = "v4_support"
AREA_TOLERANCE = 1e-6
TIMING_FIELDS = {
    "selector_wall_s", "selector_mean_wall_ms", "rng_setup_wall_s", "gate_setup_wall_s",
}
PHYSICAL_METRIC_FIELDS = (
    "arrival_horizon_s", "elapsed_including_drain_s", "job_count", "completed_jobs",
    "makespan_s", "mean_flow_s", "p90_flow_s", "time_average_unfinished_jobs",
    "frame_boundary_mean_backlog", "frame_boundary_peak_backlog", "frame_count",
    "loaded_frame_count", "maximum_expected_drift_bound_violation",
    "maximum_queue_recursion_violation", "selected_option_counts",
)
SOURCE_FILES = (
    "algorithm/experiments/aligned_lookahead_continuation.py",
    "algorithm/experiments/aligned_hardware_frame_mechanism.py",
    "algorithm/experiments/aligned_hardware_lookahead_continuation.py",
    "algorithm/experiments/aligned_hardware_rollout_depth_gate_first_divergence.py",
)


def _json(value):
    return json.loads(json.dumps(value))


def _registered_rows(development):
    rows = [
        row for row in development["runs"]
        if row["load"] == LOAD and row["seed"] == REAL_SEED and row["budget"] == BUDGET
        and (row["policy"] == V4_POLICY or row["policy"] == NEW_POLICY)
    ]
    rows.sort(key=lambda row: (row["policy"], -1 if row["planning_seed"] is None else row["planning_seed"]))
    return _json(rows)


def _deterministic_accounting(accounting):
    return {key: value for key, value in accounting.items() if key not in TIMING_FIELDS}


def _assert_gate_source_reproduced(row, original):
    for key in ("load", "seed", "budget", "policy", "planning_seed"):
        if row[key] != original[key]:
            raise ValueError(f"gated source identity changed at {key}")
    for key in ("metrics", "decomposition"):
        if _json(row[key]) != original[key]:
            raise ValueError(f"gated source {key} changed for planning seed {row['planning_seed']}")
    if _deterministic_accounting(row["selector_accounting"]) != _deterministic_accounting(original["selector_accounting"]):
        raise ValueError(f"gated source work accounting changed for planning seed {row['planning_seed']}")


def _physical_result_matches(branch, original):
    return (_json(branch["decomposition"]) == original["decomposition"]
            and all(_json(branch["metrics"][key]) == original["metrics"][key]
                    for key in PHYSICAL_METRIC_FIELDS))


def _outcome(delta_job_s):
    return ("improved" if delta_job_s < -AREA_TOLERANCE
            else "worsened" if delta_job_s > AREA_TOLERANCE else "unchanged")


def _component_delta(after, before, population):
    return {
        key: population * (after["decomposition"][key] - before["decomposition"][key])
        for key in COMPONENTS
    }


class FirstDivergenceDiscovery:
    """Preserve one native gated trajectory and retain only its first v4 difference."""

    def __init__(self, model, rates, service_rates, planning_seed, pairs):
        self.model = model
        self.native = RolloutDepthGateRecorder(
            model.options, model.lanes, rates, service_rates, planning_seed,
        )
        self.pairs = pairs
        self.prefix = []
        self.first = None
        self.inactive_prefix_frames = 0
        self.equal_active_comparisons = 0
        self.active_v4_comparisons = 0
        self.action_descriptions = 0
        self.action_description_completion_offset_reads = 0

    @property
    def frames(self):
        return self.native.frames

    def __call__(self, options, queues, lanes, budget):
        frame_index, start_s = len(self.prefix), self.native.now
        selected = self.native(options, queues, lanes, budget)
        self.prefix.append((tuple(queues), selected))
        if self.first is not None:
            return selected
        if not isinstance(selected, MixedSuccessorRolloutFrameSelection):
            self.inactive_prefix_frames += 1
            return selected

        v4 = self.pairs.support(queues)
        descriptions = tuple(action_description(self.model, action) for action in (v4, selected))
        self.active_v4_comparisons += 1
        self.action_descriptions += 2
        self.action_description_completion_offset_reads += sum(
            len(offsets) for description in descriptions for offsets in description["completion_offsets"]
        )
        if action_signature(descriptions[0]) == action_signature(descriptions[1]):
            self.equal_active_comparisons += 1
            return selected

        self.first = {
            "frame_index": frame_index,
            "start_s": start_s,
            "queue": tuple(queues),
            "weighted_backlog": self.native.gate.load(queues),
            "v4_action_object": v4,
            "gate_action_object": selected,
            "v4_action": descriptions[0],
            "gate_action": descriptions[1],
            "rollout_used": selected.rollout_used,
            "rollout_vetoed": selected.rollout_vetoed,
            "rollout_cost_delta_job_s": (
                None if selected.selected_rollout_cost is None
                else selected.selected_rollout_cost - selected.incumbent_rollout_cost
            ),
        }
        return selected

    def accounting(self):
        return {
            "inactive_prefix_frames": self.inactive_prefix_frames,
            "active_v4_comparisons": self.active_v4_comparisons,
            "equal_active_comparisons": self.equal_active_comparisons,
            "action_descriptions": self.action_descriptions,
            "action_description_completion_offset_reads": self.action_description_completion_offset_reads,
            "first_divergence_weighted_queue_coordinate_evaluations": (
                len(self.native.gate.weights) if self.first is not None else 0
            ),
            "extra_rng_initializations": 0,
            "extra_rng_state_reads": 0,
            "extra_rng_state_writes": 0,
            "extra_random_draws": 0,
        }


def _run_source(task):
    model, certificate, protocol, planning_seed, original, frozen_v4 = task
    if (planning_seed not in PLANNING_SEEDS or protocol["load"] != LOAD
            or protocol["real_seed"] != REAL_SEED or protocol["budget"] != BUDGET):
        raise ValueError("first-divergence task differs from the registered roster")
    horizon = protocol["arrival_horizon_s"]
    rates, service_rates = certificate["arrival_rates"], protocol["terminal_service_rates"]
    tape = poisson_arrivals(rates, horizon, REAL_SEED)
    pairs = DecisionPairs(model, rates, service_rates, budget=BUDGET)
    discovery = FirstDivergenceDiscovery(model, rates, service_rates, planning_seed, pairs)
    replayed = _checked_run(model, tape, horizon, certificate, discovery)
    source = {
        "load": LOAD, "seed": REAL_SEED, "budget": BUDGET,
        "policy": NEW_POLICY, "planning_seed": planning_seed,
        **replayed, "selector_accounting": discovery.native.accounting(),
    }
    _assert_gate_source_reproduced(source, original)
    discovery_search = pairs.accounting()
    base = {
        "load": LOAD, "seed": REAL_SEED, "planning_seed": planning_seed,
        "job_count": len(tape), "source_frame_count": len(discovery.frames),
        "source_reproduced": True,
        "first_divergence_count": int(discovery.first is not None),
        "replay_branch_count": 0,
        "case": None,
        "diagnostic_work": {
            "discovery": discovery.accounting(),
            "discovery_v4_cache_miss_work": discovery_search,
            "continuation_v4_cache_miss_work": dict.fromkeys(discovery_search, 0),
            "replay_runs": 0, "replay_frames": 0,
            "continuation_v4_calls": 0, "action_audit_evaluations": 0,
            "action_audit_completion_offset_reads": 0,
        },
    }
    if discovery.first is None:
        return base

    switch = discovery.first
    index, start_s, queues = switch["frame_index"], switch["start_s"], switch["queue"]
    actions = (switch["v4_action_object"], switch["gate_action_object"])
    times = tuple(time for time, _, _ in tape)
    sums = tuple(accumulate(times, initial=0.0))
    class_times = [
        tuple(time for time, coordinate, _ in tape if coordinate == index)
        for index in range(len(model.workloads))
    ]
    audits = [
        action_audit(model, queues, action, start_s, horizon, rates, service_rates,
                     times, sums, class_times)
        for action in actions
    ]
    action_audit_offset_reads = 2 * sum(
        len(offsets) for description in (switch["v4_action"], switch["gate_action"])
        for offsets in description["completion_offsets"]
    )
    branches, recorders = [], []
    for action in actions:
        recorder = PrefixReplay(discovery.prefix, index, action, pairs.support)
        recorders.append(recorder)
        branches.append(_checked_run(model, tape, horizon, certificate, recorder))
    if not _physical_result_matches(branches[0], frozen_v4):
        raise ValueError("v4-first common continuation does not reproduce frozen v4 physically")
    effect = paired_effect(branches[0], branches[1], audits, len(tape))
    if (abs(effect["residual_identity_error_job_s"]) > 1e-5
            or abs(effect["flow_component_identity_error_job_s"]) > 1e-5):
        raise ValueError("first-divergence paired area decomposition failed")

    full_total = len(tape) * (
        original["metrics"]["mean_flow_s"] - frozen_v4["metrics"]["mean_flow_s"]
    )
    suffix_total = len(tape) * (
        original["metrics"]["mean_flow_s"] - branches[1]["metrics"]["mean_flow_s"]
    )
    total_identity_error = full_total - effect["realized_total_delta_job_s"] - suffix_total
    full_components = _component_delta(original, frozen_v4, len(tape))
    suffix_components = _component_delta(original, branches[1], len(tape))
    component_identity_errors = {
        key: full_components[key] - effect["flow_component_deltas_job_s"][key] - suffix_components[key]
        for key in COMPONENTS
    }
    if abs(total_identity_error) > 1e-5 or max(map(abs, component_identity_errors.values())) > 1e-5:
        raise ValueError("full-policy first-action suffix identity failed")

    final_search = pairs.accounting()
    continuation_search = {key: final_search[key] - discovery_search[key] for key in final_search}
    continuation_calls = sum(len(recorder.frames) - index - 1 for recorder in recorders)
    metadata = {
        key: value for key, value in switch.items()
        if key not in ("v4_action_object", "gate_action_object")
    }
    metadata["phase"] = (
        "drain" if start_s >= horizon else "crosses_cutoff"
        if start_s + max(action.duration for action in actions) > horizon else "arrival_window"
    )
    metadata["prefix_accounting_identity"] = (
        index == discovery.inactive_prefix_frames + discovery.equal_active_comparisons
    )
    metadata["v4_branch_matches_frozen_physical_result"] = True
    metadata["branch_mean_flow_s"] = {
        "v4_first_then_v4": branches[0]["metrics"]["mean_flow_s"],
        "gate_first_then_v4": branches[1]["metrics"]["mean_flow_s"],
    }
    metadata["first_action_effect"] = effect
    metadata["full_gate_minus_v4_effect"] = {
        "realized_total_delta_job_s": full_total,
        "mean_flow_delta_s": full_total / len(tape),
        "flow_component_deltas_job_s": full_components,
        "outcome": _outcome(full_total),
    }
    metadata["post_first_gated_suffix_effect"] = {
        "realized_total_delta_job_s": suffix_total,
        "mean_flow_delta_s": suffix_total / len(tape),
        "flow_component_deltas_job_s": suffix_components,
        "outcome": _outcome(suffix_total),
    }
    metadata["full_first_suffix_identity_error_job_s"] = total_identity_error
    metadata["flow_component_identity_errors_job_s"] = component_identity_errors
    return {
        **base, "replay_branch_count": 2, "case": metadata,
        "diagnostic_work": {
            "discovery": discovery.accounting(),
            "discovery_v4_cache_miss_work": discovery_search,
            "continuation_v4_cache_miss_work": continuation_search,
            "replay_runs": 2,
            "replay_frames": sum(len(recorder.frames) for recorder in recorders),
            "continuation_v4_calls": continuation_calls,
            "action_audit_evaluations": 2,
            "action_audit_completion_offset_reads": action_audit_offset_reads,
        },
    }


def _source_snapshot():
    sources = json.loads(DEVELOPMENT_SOURCES.read_text())
    if any((ROOT / name).read_text() != text for name, text in sources.items()):
        raise ValueError("registered depth-gate development sources changed")
    return {**sources, **{name: (ROOT / name).read_text() for name in SOURCE_FILES}}


def register(output=DEFAULT_OUTPUT):
    output = Path(output)
    protocol_path = output.with_name(output.stem + "_protocol.json")
    snapshot_path = output.with_name(output.stem + "_fixed_sources.json")
    if any(path.exists() for path in (output, protocol_path, snapshot_path)):
        raise ValueError("first-divergence artifacts already exist")
    development = json.loads(DEVELOPMENT.read_text())
    if (development["status"] != "PASS" or not all(development["checks"].values())
            or development["development_advance"]):
        raise ValueError("first-divergence diagnosis requires the failed, executed depth-gate development")
    expected_advance_checks = {
        "execution_passed": True,
        "mean_flow_and_p90_no_worse_than_v4_at_each_load": True,
        "every_medium_high_real_seed_improves_flow_over_v4": False,
    }
    failed_cells = [
        (summary["load"], seed["seed"])
        for summary in development["summary_by_load"] if summary["load"] in (.8, .95)
        for seed in summary["by_arrival_seed"]
        if seed["flow_ratio_over_v4_support"] >= 1.0
    ]
    if (development["development_advance_checks"] != expected_advance_checks
            or failed_cells != [(LOAD, REAL_SEED)]):
        raise ValueError("load 0.8 seed 41 is not the unique registered medium/high-load failure")
    rows = _registered_rows(development)
    if (len(rows) != 4
            or [(row["policy"], row["planning_seed"]) for row in rows]
            != [(NEW_POLICY, 0), (NEW_POLICY, 1), (NEW_POLICY, 2), (V4_POLICY, None)]):
        raise ValueError("diagnostic source roster must contain three gated rows and one v4 row")
    model = load_model()
    certificate = capacity_certificate(model, LOAD)
    service_rates = terminal_service_rates(model, certificate)
    source_protocol = development["protocol"]
    if (source_protocol["loads"] != [.5, .8, .95]
            or source_protocol["seeds"] != [41, 42, 43]
            or source_protocol["planning_seeds"] != [0, 1, 2]
            or source_protocol["budget"] != BUDGET
            or source_protocol["gate_weights"] != list(GATE_WEIGHTS)
            or source_protocol["gate_capacity"] != GATE_CAPACITY
            or source_protocol["strict_threshold"] != GATE_THRESHOLD
            or _json(model.snapshot()) != source_protocol["model"]
            or _json(certificate) != source_protocol["certificates"][1]
            or list(service_rates) != source_protocol["terminal_service_rates"]):
        raise ValueError("diagnostic model, rates, roster or gate differs from its development source")
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "development failure diagnosis on consumed data only",
        "source_experiment": DEVELOPMENT.name,
        "selected_failed_cell": {"load": LOAD, "real_seed": REAL_SEED},
        "load": LOAD, "real_seed": REAL_SEED, "planning_seeds": PLANNING_SEEDS,
        "budget": BUDGET, "arrival_horizon_s": source_protocol["arrival_horizon_s"],
        "terminal_service_rates": service_rates,
        "model": model.snapshot(), "certificate": certificate,
        "gate_weights": GATE_WEIGHTS, "gate_capacity": GATE_CAPACITY,
        "strict_threshold": GATE_THRESHOLD,
        "source_rows": rows,
        "state_roster": "the chronologically first physical gate/v4 action difference on each of all three gated source trajectories; retain a zero-case source if none exists",
        "physical_action": "departures, duration and sorted per-class completion offsets; lane permutations are equal",
        "intervention": "replay the identical gated prefix; force v4 versus gate once; then use memoized frozen v4 K26 in both branches until the identical original population drains",
        "sign": "all effects are gate-first minus v4-first in job-seconds",
        "decomposition": "full gate-v4 = first action under common v4 + remaining full-gate versus gate-once suffix, for total flow and each flow component",
        "diagnostic_work": "native source work, deterministic v4 shadow cache misses, action audits, replay frames and common-v4 cache misses are reported separately; shadow and continuation add no random draws",
        "stopping_rule": "three registered source trajectories once, at most one first divergence and two replay branches per source; no threshold fitting, extra states, extra seeds, repeated planning samples or confirmation data",
        "interpretation": "local first-action direction and later gated-policy feedback only; correlated planning runs on one arrival tape are not independent replicates",
    }
    sources = _source_snapshot()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_new(protocol_path, protocol)
    _write_new(snapshot_path, sources)
    return protocol


def _execution_checks(runs, sources):
    cases = [row["case"] for row in runs if row["case"] is not None]
    discovered = sum(row["first_divergence_count"] for row in runs)
    return {
        "three_registered_gated_sources_reproduced": (
            len(runs) == 3
            and sorted(row["planning_seed"] for row in runs) == list(PLANNING_SEEDS)
            and all(row["source_reproduced"] for row in runs)
        ),
        "at_most_one_first_physical_divergence_per_source": (
            all(
                row["first_divergence_count"] in (0, 1)
                and (row["case"] is not None) == bool(row["first_divergence_count"])
                for row in runs
            )
            and len(cases) == discovered
        ),
        "first_divergence_scan_stops_at_first_or_exhausts_source": all(
            (
                row["diagnostic_work"]["discovery"]["active_v4_comparisons"]
                == row["diagnostic_work"]["discovery"]["equal_active_comparisons"] + 1
                and row["case"]["frame_index"]
                == row["diagnostic_work"]["discovery"]["inactive_prefix_frames"]
                + row["diagnostic_work"]["discovery"]["equal_active_comparisons"]
            ) if row["case"] is not None else (
                row["diagnostic_work"]["discovery"]["active_v4_comparisons"]
                == row["diagnostic_work"]["discovery"]["equal_active_comparisons"]
                and row["source_frame_count"]
                == row["diagnostic_work"]["discovery"]["inactive_prefix_frames"]
                + row["diagnostic_work"]["discovery"]["equal_active_comparisons"]
            )
            for row in runs
        ),
        "first_divergence_prefix_accounting_holds": all(case["prefix_accounting_identity"] for case in cases),
        "first_divergences_are_active_and_strict": all(case["weighted_backlog"] > GATE_THRESHOLD for case in cases),
        "first_divergence_actions_are_physically_distinct": all(action_signature(case["v4_action"]) != action_signature(case["gate_action"]) for case in cases),
        "two_common_v4_replay_branches_per_divergence": (
            all(
                row["replay_branch_count"]
                == row["diagnostic_work"]["replay_runs"]
                == 2 * row["first_divergence_count"]
                for row in runs
            )
            and sum(row["replay_branch_count"] for row in runs) == 2 * len(cases)
        ),
        "v4_first_branches_reproduce_frozen_v4_physically": all(case["v4_branch_matches_frozen_physical_result"] for case in cases),
        "paired_area_and_flow_component_identities_hold": all(abs(case["first_action_effect"]["residual_identity_error_job_s"]) <= 1e-5 and abs(case["first_action_effect"]["flow_component_identity_error_job_s"]) <= 1e-5 for case in cases),
        "full_first_suffix_identities_hold": all(abs(case["full_first_suffix_identity_error_job_s"]) <= 1e-5 and max(map(abs, case["flow_component_identity_errors_job_s"].values())) <= 1e-5 for case in cases),
        "diagnostic_work_accounting_holds": all(
            row["diagnostic_work"]["discovery"]["first_divergence_weighted_queue_coordinate_evaluations"]
            == len(GATE_WEIGHTS) * row["first_divergence_count"]
            and row["diagnostic_work"]["action_audit_evaluations"] == 2 * row["first_divergence_count"]
            and row["diagnostic_work"]["action_audit_completion_offset_reads"]
            == (0 if row["case"] is None else 2 * sum(
                len(offsets)
                for action in (row["case"]["v4_action"], row["case"]["gate_action"])
                for offsets in action["completion_offsets"]
            ))
            for row in runs
        ),
        "diagnostic_shadow_and_continuation_add_no_randomness": all(all(row["diagnostic_work"]["discovery"][key] == 0 for key in ("extra_rng_initializations", "extra_rng_state_reads", "extra_rng_state_writes", "extra_random_draws")) for row in runs),
        "common_continuation_uses_only_v4": all(
            row["diagnostic_work"]["discovery_v4_cache_miss_work"]["lookahead_calls"] == 0
            and row["diagnostic_work"]["continuation_v4_cache_miss_work"]["lookahead_calls"] == 0
            for row in runs
        ),
        "registered_sources_unchanged": all((ROOT / name).read_text() == text for name, text in sources.items()),
    }


def run_registered(output=DEFAULT_OUTPUT):
    output = Path(output)
    protocol_path = output.with_name(output.stem + "_protocol.json")
    snapshot_path = output.with_name(output.stem + "_fixed_sources.json")
    if output.exists():
        raise ValueError("first-divergence result already exists")
    protocol = json.loads(protocol_path.read_text())
    sources = json.loads(snapshot_path.read_text())
    expected = {
        "load": LOAD, "real_seed": REAL_SEED, "planning_seeds": list(PLANNING_SEEDS),
        "budget": BUDGET, "gate_weights": list(GATE_WEIGHTS),
        "gate_capacity": GATE_CAPACITY, "strict_threshold": GATE_THRESHOLD,
    }
    if any(protocol[key] != value for key, value in expected.items()):
        raise ValueError("registered first-divergence roster or gate changed")
    if any((ROOT / name).read_text() != text for name, text in sources.items()):
        raise ValueError("registered first-divergence sources changed")
    development = json.loads(DEVELOPMENT.read_text())
    if _registered_rows(development) != protocol["source_rows"]:
        raise ValueError("registered first-divergence source rows changed")
    model = load_model()
    certificate = capacity_certificate(model, LOAD)
    if (_json(model.snapshot()) != protocol["model"]
            or _json(certificate) != protocol["certificate"]
            or list(terminal_service_rates(model, certificate)) != protocol["terminal_service_rates"]):
        raise ValueError("registered first-divergence model or rates changed")
    gated = {row["planning_seed"]: row for row in protocol["source_rows"] if row["policy"] == NEW_POLICY}
    frozen_v4 = next(row for row in protocol["source_rows"] if row["policy"] == V4_POLICY)
    runs = [
        _run_source((model, certificate, protocol, planning_seed, gated[planning_seed], frozen_v4))
        for planning_seed in PLANNING_SEEDS
    ]
    checks = _execution_checks(runs, sources)
    report = {
        "experiment": "aligned_hardware_rollout_depth_gate_first_divergence_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks, "protocol": protocol, "runs": runs,
        "summary_by_planning_seed": [
            {
                "planning_seed": row["planning_seed"],
                "frame_index": row["case"]["frame_index"] if row["case"] else None,
                "start_s": row["case"]["start_s"] if row["case"] else None,
                "queue": row["case"]["queue"] if row["case"] else None,
                "weighted_backlog": row["case"]["weighted_backlog"] if row["case"] else None,
                "first_action_effect": row["case"]["first_action_effect"] if row["case"] else None,
                "full_gate_minus_v4_effect": row["case"]["full_gate_minus_v4_effect"] if row["case"] else None,
                "post_first_gated_suffix_effect": row["case"]["post_first_gated_suffix_effect"] if row["case"] else None,
            }
            for row in runs
        ],
        "limitations": [
            "The first-action intervention is local to one policy-induced state and cannot reconstruct later gate decisions by itself.",
            "The post-first suffix combines later gate choices with their state feedback; it is not a pure state-feedback effect.",
            "All planning seeds share one consumed real-arrival tape and are correlated descriptive cases, not independent replicates.",
        ],
    }
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
