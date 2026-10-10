"""Summarize the small completed fixed-three common-horizon ranking artifact."""
import argparse
from fractions import Fraction
import json
from pathlib import Path

FIELDS = ("cost", "common_area", "waiting_area", "inflight_area", "terminal_cost")
STEM = "aligned_hardware_common_horizon_flow_20261003_r1"
ARTIFACTS = Path(__file__).resolve().parents[2] / "md" / "experiment_artifacts"


def exact(encoded):
    return Fraction(*encoded)


def preference(value):
    return "parent" if value < 0 else "native" if value > 0 else "tie"


def analyze(summary):
    records = []
    for state in summary["records"]:
        comparison = state["comparison"]
        row = dict(seed=state["seed"], tape_id=state["tape_id"],
            available=comparison["available"], root_preference=comparison["S1_preference"],
            previous_preferences={name: old["S1_preference"]
                for name, old in state["previous_comparisons"].items()},
            actual_finite_cohort_context=state["birth_cohort_root_effects"]["finite"],
            failed_samples={role: root["failed_samples"] for role, root in state["roots"].items()},
            overflow_samples={role: root["overflow_samples"] for role, root in state["roots"].items()})
        if not comparison["available"]:
            records.append(row)
            continue
        delta = {field: exact(comparison["exact_parent_minus_native_mean"][field]) for field in FIELDS}
        values = [exact(sample["exact_parent_minus_native"]["cost"]) for sample in state["samples"]]
        if len(values) != 8 or not all(sample["available"] for sample in state["samples"]):
            raise ValueError("an available state must retain all eight complete paired samples")
        total = sum(values, Fraction(0))
        if total / 8 != delta["cost"] or delta["cost"] != delta["common_area"] + delta["terminal_cost"]:
            raise ValueError("sample means or common/terminal components do not reconstruct the score")
        if delta["common_area"] != delta["waiting_area"] + delta["inflight_area"]:
            raise ValueError("waiting and service-in-progress components do not reconstruct common area")
        population = comparison["exact_population_parent_minus_native_mean"]
        sources = {source: {field: exact(group[field]) for field in FIELDS}
            for source, group in population["sources"].items()}
        classes = [{"coordinate": group["coordinate"],
            **{field: exact(group[field]) for field in FIELDS}}
            for group in population["per_coordinate"]]
        for field in FIELDS:
            if sum((group[field] for group in sources.values()), Fraction(0)) != delta[field]:
                raise ValueError("source deltas do not reconstruct the same score")
            if sum((group[field] for group in classes), Fraction(0)) != delta[field]:
                raise ValueError("class deltas do not reconstruct the same score")
        leave_one_out = [(total - value) / 7 for value in values]
        row.update(parent_minus_native_mean={field: float(value) for field, value in delta.items()},
            sample_cost_deltas=[float(value) for value in values],
            sample_preferences={name: sum(preference(value) == name for value in values)
                for name in ("parent", "native", "tie")},
            leave_one_out_mean_range=[float(min(leave_one_out)), float(max(leave_one_out))],
            leave_one_out_preferences=[preference(value) for value in leave_one_out],
            mean_preference_retained_after_any_one_sample_removal=all(
                preference(value) == preference(delta["cost"]) for value in leave_one_out),
            source_parent_minus_native_mean={source: {field: float(value) for field, value in group.items()}
                for source, group in sources.items()},
            per_coordinate_parent_minus_native_mean=[{"coordinate": group["coordinate"],
                **{field: float(group[field]) for field in FIELDS}} for group in classes],
            cnn_cost_delta=float(next(group["cost"] for group in classes if group["coordinate"] == 0)))
        records.append(row)
    return dict(registered_at_utc=summary["registered_at_utc"],
        execution_checks_passed=summary["execution_checks_passed"],
        available_state_rankings=summary["available_state_rankings"], records=records,
        performance_gain_established=False, full24_decision=summary["full24_decision"],
        interpretation="Conditional fixed-state action ranking only. Leave-one-out describes the original eight samples; it does not add samples or set a promotion gate. Actual cohort outcomes are contextual labels from earlier finite continuations, not the evaluated scoring policy. Full-policy performance remains unestablished.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=ARTIFACTS / (STEM + "_summary.json"))
    parser.add_argument("--output", type=Path, default=ARTIFACTS / (STEM + "_analysis.json"))
    args = parser.parse_args()
    result = analyze(json.loads(args.summary.read_text()))
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    for row in result["records"]:
        print(json.dumps(row, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
