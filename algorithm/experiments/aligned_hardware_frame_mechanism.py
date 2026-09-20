"""Development-only flow decomposition and controlled v4 action substitutions.

Uses seeds 41--43 and the frozen simulator. Exact shadow decisions are offline
diagnostic work; the intervention policies are not budget-26 algorithms.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean

from algorithm.budgeted_frame_candidates import select_budgeted_frame
from algorithm.experiments.aligned_frame_flow_decomposition import decompose_flow
from algorithm.experiments.aligned_hardware_frame_experiment import poisson_arrivals, simulate_frames
from algorithm.experiments.aligned_hardware_frame_model import capacity_certificate, load_model


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACTS / "aligned_hardware_frame_mechanism_development_20260909.json"
MODES = ("exact", "k26", "k26_on_strict_loss", "k26_on_equal_score", "first_difference_once")
FLOW_PARTS = ("mean_deferred_visible_wait_s", "mean_arrival_barrier_wait_s", "mean_dispatch_to_completion_s")
SCORE_TOLERANCE = 1e-10


def action_description(model, action):
    offsets = tuple(
        tuple(sorted(offset for index in action.option_indices if index is not None
                     for offset in model.options[index].completion_offsets[coordinate]))
        for coordinate in range(len(model.workloads))
    )
    return {
        "option_indices": action.option_indices, "departures": action.service,
        "duration_s": action.duration, "score": action.score,
        "completion_offsets": offsets,
        "dispatch_to_completion_sum_s": sum(map(sum, offsets)),
        "loaded": any(model.options[index].loaded for index in action.option_indices if index is not None),
    }


def action_signature(description):
    # Pure lane permutations cannot change total mean flow for a count-only
    # policy. They may reassign completion offsets to individual FIFO jobs.
    return description["departures"], description["duration_s"], description["completion_offsets"]


class DecisionPairs:
    """Memoize deterministic shadow decisions by Q, never by future arrivals."""
    def __init__(self, model):
        self.model = model
        self.cache = {}
        self.search_attempts = self.option_probes = self.threshold_queries = 0

    def get(self, queues):
        key = tuple(queues)
        if key not in self.cache:
            small = select_budgeted_frame(self.model.options, key, self.model.lanes, 26)
            exact = select_budgeted_frame(self.model.options, key, self.model.lanes, 120)
            if not exact.exact or small.score > exact.score + SCORE_TOLERANCE:
                raise ValueError("diagnostic exact reference failed to bound K26")
            descriptions = (action_description(self.model, small), action_description(self.model, exact))
            self.cache[key] = small, exact, descriptions
            for result in (small, exact):
                self.search_attempts += result.attempts
                self.option_probes += result.bound_option_probes
                self.threshold_queries += result.bound_threshold_evaluations
        return self.cache[key]


class Recorder:
    def __init__(self, pairs, mode):
        if mode not in MODES:
            raise ValueError(f"unknown diagnostic mode: {mode}")
        self.pairs, self.mode = pairs, mode
        self.now = 0.0
        self.frames = []
        self.intervention = None
        self.substituted_frames = 0

    def __call__(self, options, queues, lanes, budget):
        small, exact, (small_desc, exact_desc) = self.pairs.get(queues)
        strict = exact.score - small.score > SCORE_TOLERANCE
        different = action_signature(small_desc) != action_signature(exact_desc)
        choose_small = (
            self.mode == "k26"
            or (self.mode == "k26_on_strict_loss" and strict)
            or (self.mode == "k26_on_equal_score" and not strict)
        )
        if self.mode == "first_difference_once" and different and self.intervention is None:
            choose_small = True
            self.intervention = {
                "frame_index": len(self.frames), "start_s": self.now, "queue": tuple(queues),
                "k26": small_desc, "exact": exact_desc,
                "support_gap": exact.score - small.score,
            }
        chosen = small if choose_small else exact
        self.substituted_frames += int(choose_small and different)
        self.frames.append({
            "start_s": self.now, "duration_s": chosen.duration,
            "queue_before": tuple(queues), "departures": chosen.service,
            "option_indices": chosen.option_indices,
        })
        self.now += chosen.duration
        return chosen


def shadow_summary(model, frames, pairs):
    groups = defaultdict(lambda: {"frame_count": 0, "visited_time_s": 0.0, "support_gap_sum": 0.0,
                                  "duration_delta_sum_s": 0.0, "departures_delta_sum": 0})
    different_states = Counter()
    for frame in frames:
        small, exact, (small_desc, exact_desc) = pairs.get(frame["queue_before"])
        if action_signature(small_desc) == action_signature(exact_desc):
            category = "same_mean_flow_action"
        elif exact.score - small.score <= SCORE_TOLERANCE:
            category = "equal_score_different_action"
        else:
            duration_delta = small.duration - exact.duration
            category = "strict_loss_" + ("shorter" if duration_delta < 0 else "longer" if duration_delta > 0 else "same_duration")
        row = groups[category]
        row["frame_count"] += 1
        row["visited_time_s"] += frame["duration_s"]
        row["support_gap_sum"] += exact.score - small.score
        row["duration_delta_sum_s"] += small.duration - exact.duration
        row["departures_delta_sum"] += sum(small.service) - sum(exact.service)
        if category != "same_mean_flow_action":
            different_states[frame["queue_before"]] += 1
    total_time = sum(frame["duration_s"] for frame in frames)
    examples = []
    for queues, count in different_states.most_common(6):
        small, exact, descriptions = pairs.get(queues)
        examples.append({"queue": queues, "visits": count, "k26": descriptions[0], "exact": descriptions[1]})
    return {
        "population": "states visited by this trajectory; not pooled with the other policy",
        "categories": {key: dict(value, time_fraction=value["visited_time_s"] / total_time) for key, value in groups.items()},
        "most_frequent_different_states": examples,
    }


def _expected_barrier_area(frames, horizon, rates):
    total_rate = sum(rates)
    return total_rate * sum(
        frame["duration_s"] * exposure - exposure * exposure / 2
        for frame in frames
        for exposure in (max(0.0, min(frame["duration_s"], horizon - frame["start_s"])),)
    )


def _assert_frozen_sources():
    snapshot = json.loads((ARTIFACTS / "aligned_hardware_frame_confirmation_v4_20260909/source_snapshot.json").read_text())
    changed = [name for name, source in snapshot.items() if (ROOT / name).read_text() != source]
    if changed:
        raise ValueError(f"frozen source changed: {changed}")


def build_diagnosis(output=DEFAULT_OUTPUT):
    _assert_frozen_sources()
    development = json.loads((ARTIFACTS / "aligned_hardware_frame_experiment_v4_20260909.json").read_text())
    protocol = {
        "role": "development mechanism diagnosis; no independent confirmation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": [41, 42, 43], "loads": [0.50, 0.80, 0.95], "arrival_horizon_s": 20_000.0,
        "small_budget": 26, "exact_budget": 120, "modes": MODES,
        "score_classification_tolerance": SCORE_TOLERANCE,
        "interventions": {
            "k26_on_strict_loss": "use K26 when its support is below exact; use exact otherwise",
            "k26_on_equal_score": "use K26 on equal-support decisions; use exact otherwise",
            "first_difference_once": "replace only the first mean-flow-relevant action difference on the shared exact prefix, then return to exact",
        },
        "primary_accounting": "arrival barrier wait + deferred visible queue wait + dispatch-to-completion time, over all jobs including drain",
        "diagnostic_cost": "both decisions computed offline and cached by queue; intervention modes are not budget-26 schedulers",
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_name(output.stem + "_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    model = load_model()
    if json.loads(json.dumps(model.snapshot())) != development["model"]:
        raise ValueError("diagnostic model differs from the frozen development model")
    pairs = DecisionPairs(model)
    runs, comparisons = [], []
    for load in protocol["loads"]:
        certificate = capacity_certificate(model, load)
        for seed in protocol["seeds"]:
            tape = poisson_arrivals(certificate["arrival_rates"], protocol["arrival_horizon_s"], seed)
            local = {}
            for mode in MODES:
                recorder = Recorder(pairs, mode)
                budget = 26 if mode == "k26" else 120
                metrics = simulate_frames(model, tape, protocol["arrival_horizon_s"], certificate, budget, selector=recorder)
                decomposition = decompose_flow(model, tape, recorder.frames, protocol["arrival_horizon_s"])
                error = decomposition["mean_flow_s"] - metrics["mean_flow_s"]
                if abs(error) > 1e-8 or decomposition["completed_jobs"] != metrics["completed_jobs"]:
                    raise ValueError("diagnostic decomposition does not reconstruct the frozen simulator")
                if mode in ("k26", "exact"):
                    original = next(row for row in development["runs"] if (row["load"], row["seed"], row["budget"]) == (load, seed, budget))
                    normalized = json.loads(json.dumps(metrics))
                    changed = [key for key, value in normalized.items() if key in original and original[key] != value]
                    if changed:
                        raise ValueError(f"diagnostic recording changed frozen {mode} run fields: {changed}")
                record = {
                    "load": load, "seed": seed, "mode": mode, "metrics": metrics,
                    "decomposition": decomposition, "reconstruction_error_s": error,
                    "expected_conditional_arrival_barrier_area_s": _expected_barrier_area(recorder.frames, protocol["arrival_horizon_s"], certificate["arrival_rates"]),
                    "substituted_frames": recorder.substituted_frames, "first_intervention": recorder.intervention,
                }
                if mode in ("k26", "exact"):
                    record["same_state_shadow"] = shadow_summary(model, recorder.frames, pairs)
                local[mode] = record
                runs.append(record)
            reference = local["exact"]
            for mode, row in local.items():
                components = {key: row["decomposition"][key] - reference["decomposition"][key] for key in FLOW_PARTS}
                delta = row["metrics"]["mean_flow_s"] - reference["metrics"]["mean_flow_s"]
                comparisons.append({
                    "load": load, "seed": seed, "mode": mode,
                    "mean_flow_delta_s": delta,
                    "mean_flow_ratio": row["metrics"]["mean_flow_s"] / reference["metrics"]["mean_flow_s"],
                    "mean_flow_component_deltas_s": components,
                    "component_identity_error_s": sum(components.values()) - delta,
                    "makespan_delta_s": row["metrics"]["makespan_s"] - reference["metrics"]["makespan_s"],
                })
            print(json.dumps({"load": load, "seed": seed, "modes_completed": len(MODES)}), flush=True)
    _assert_frozen_sources()
    grouped = []
    for load in protocol["loads"]:
        for mode in MODES:
            rows = [row for row in comparisons if row["load"] == load and row["mode"] == mode]
            grouped.append({
                "load": load, "mode": mode,
                "mean_paired_flow_delta_s": mean(row["mean_flow_delta_s"] for row in rows),
                "paired_flow_delta_range_s": [min(row["mean_flow_delta_s"] for row in rows), max(row["mean_flow_delta_s"] for row in rows)],
                "improved_seed_count": sum(row["mean_flow_delta_s"] < -1e-10 for row in rows),
                "mean_component_deltas_s": {key: mean(row["mean_flow_component_deltas_s"][key] for row in rows) for key in FLOW_PARTS},
            })
    checks = {
        "frozen_baseline_trajectories_reproduced": True,
        "flow_decomposition_matches_simulator": all(abs(row["reconstruction_error_s"]) < 1e-8 for row in runs),
        "paired_component_identity": all(abs(row["component_identity_error_s"]) < 1e-8 for row in comparisons),
        "all_jobs_complete": all(row["metrics"]["job_count"] == row["metrics"]["completed_jobs"] for row in runs),
        "frozen_sources_unchanged": True,
    }
    result = {
        "experiment": "aligned_frame_flow_mechanism_development", "status": "PASS" if all(checks.values()) else "FAIL",
        "protocol": protocol, "model": model.snapshot(), "checks": checks, "runs": runs, "paired_comparisons": comparisons,
        "summary_by_load_and_mode": grouped,
        "offline_shadow_work": {"unique_queue_states": len(pairs.cache), "selector_calls": 2 * len(pairs.cache),
                                "candidate_attempts": pairs.search_attempts, "option_probes": pairs.option_probes,
                                "threshold_queries": pairs.threshold_queries},
        "limitations": [
            "Development seeds and a deterministic cached phase model support mechanism diagnosis, not an additional confirmation or physical-hardware claim.",
            "Exact means maximum per-frame support, not optimal long-run mean flow.",
            "Accounting components are exact identities, not separately identified causal effects.",
            "Switch interventions identify the total effect of their complete action-selection rule; they do not isolate duration, loaded placement, or service count individually.",
            "Strict-loss-only and equal-score-only interventions visit different endogenous queues; their effects cannot be added to decompose the full K26 policy effect.",
            "The first-difference intervention changes one full action and follows exact afterwards; its effect cannot be summed to explain the full K26 trajectory.",
            "Per-run search counters replay the frozen decision accounting; offline_shadow_work counts the actual cached selected-plus-shadow queries. These counters are not additive runtime costs.",
        ],
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build_diagnosis(args.output)
    print(json.dumps({"status": result["status"], "runs": len(result["runs"]), "output": str(args.output)}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
