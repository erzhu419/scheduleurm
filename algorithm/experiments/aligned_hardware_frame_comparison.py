"""Compare paired aligned-frame experiments and their own exact references.

Run: python3 -m algorithm.experiments.aligned_hardware_frame_comparison
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping


ARTIFACT_ROOT = Path(__file__).resolve().parents[2] / "md" / "experiment_artifacts"
DEFAULT_BASELINE = ARTIFACT_ROOT / "aligned_hardware_frame_experiment_20260909.json"
DEFAULT_CURRENT = ARTIFACT_ROOT / "aligned_hardware_frame_experiment_v4_20260909.json"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "aligned_hardware_frame_comparison_v4_20260909.json"
SECTIONS = (
    "baseline", "current", "current_over_baseline_same_budget",
    "baseline_over_own_exact", "current_over_own_exact",
)


def _indexed_runs(report: Mapping[str, Any], label: str) -> dict:
    if report["status"] != "PASS":
        raise ValueError(f"{label}: input experiment did not PASS")
    indexed = {}
    for row in report["runs"]:
        key = (row["load"], row["seed"], row["budget"])
        if key in indexed:
            raise ValueError(f"{label}: duplicate (load, seed, budget) {key}")
        if row["completed_jobs"] != row["job_count"]:
            raise ValueError(f"{label}: incomplete job population at {key}")
        indexed[key] = row
    protocol = report["protocol"]
    expected = {
        (load, seed, budget)
        for load in protocol["loads"]
        for seed in protocol["seeds"]
        for budget in protocol["budgets"]
    }
    if not expected or set(indexed) != expected:
        raise ValueError(f"{label}: run keys do not match the registered protocol matrix")
    for load in protocol["loads"]:
        for seed in protocol["seeds"]:
            counts = {indexed[(load, seed, budget)]["job_count"] for budget in protocol["budgets"]}
            if len(counts) != 1:
                raise ValueError(f"{label}: budgets have different job counts at {(load, seed)}")
            reference = indexed[(load, seed, protocol["exact_reference_budget"])]
            if reference["exact_selection_frame_count"] != reference["frame_count"]:
                raise ValueError(f"{label}: reference search was not exact at {(load, seed)}")
    return indexed


def _metrics(row: Mapping[str, Any]) -> dict[str, float | None]:
    frames = row["frame_count"]
    metrics = {
        "mean_flow_s": row["mean_flow_s"],
        "p90_flow_s": row["p90_flow_s"],
        "candidate_attempts_per_frame": row["candidate_attempts"] / frames,
        "bound_option_probes_per_frame": row["bound_option_probes"] / frames,
    }
    if "bound_threshold_evaluations" in row:
        metrics["bound_threshold_evaluations_per_frame"] = row["bound_threshold_evaluations"] / frames
    return metrics


def _ratios(numerator: Mapping, denominator: Mapping) -> dict[str, float | None]:
    return {
        key: numerator[key] / denominator[key]
        if numerator[key] is not None and denominator[key] not in (None, 0) else None
        for key in numerator if key in denominator
    }


def _seed_range(values: list[float | None]) -> dict[str, float | int | None]:
    observed = [value for value in values if value is not None]
    return {
        "mean": mean(observed) if observed else None,
        "min": min(observed) if observed else None,
        "max": max(observed) if observed else None,
        "seed_count": len(observed),
    }


def compare_reports(baseline: Mapping[str, Any], current: Mapping[str, Any]) -> dict:
    """Refuse changed experiments before computing any performance ratios."""
    for field in ("model", "protocol", "certificates"):
        if baseline[field] != current[field]:
            raise ValueError(f"{field} differs between baseline and current; paired comparison refused")
    old = _indexed_runs(baseline, "baseline")
    new = _indexed_runs(current, "current")
    if old.keys() != new.keys():
        raise ValueError("baseline and current (load, seed, budget) keys differ")
    for key in old:
        if old[key]["job_count"] != new[key]["job_count"]:
            raise ValueError(f"paired job counts differ at {key}")

    exact_budget = baseline["protocol"]["exact_reference_budget"]
    pairs = []
    for load, seed, budget in sorted(old):
        key = (load, seed, budget)
        reference_key = (load, seed, exact_budget)
        before = _metrics(old[key])
        after = _metrics(new[key])
        pairs.append({
            "load": load, "seed": seed, "budget": budget,
            "job_count": old[key]["job_count"],
            "baseline_frame_count": old[key]["frame_count"],
            "current_frame_count": new[key]["frame_count"],
            "baseline": before,
            "current": after,
            "current_over_baseline_same_budget": _ratios(after, before),
            "baseline_over_own_exact": _ratios(before, _metrics(old[reference_key])),
            "current_over_own_exact": _ratios(after, _metrics(new[reference_key])),
        })
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for pair in pairs:
        grouped[(pair["load"], pair["budget"])].append(pair)
    summary = []
    for (load, budget), rows in sorted(grouped.items()):
        row = {"load": load, "budget": budget, "seeds": [pair["seed"] for pair in rows]}
        for section in SECTIONS:
            metric_keys = sorted({key for pair in rows for key in pair[section]})
            row[section] = {
                key: _seed_range([pair[section].get(key) for pair in rows])
                for key in metric_keys
            }
        summary.append(row)
    return {
        "status": "PASS",
        "checks": {
            "model_protocol_certificates_identical": True,
            "registered_run_matrices_paired": True,
            "paired_job_populations_identical_and_complete": True,
            "each_version_reference_is_exact": True,
        },
        "exact_reference_budget": exact_budget,
        "paired_run_count": len(pairs),
        "paired_comparisons": pairs,
        "summary_by_load_and_budget": summary,
        "aggregation": "arithmetic mean of seed-level values or paired ratios; min and max across seeds",
        "cost_unit": "selector operations per frame; count ratios are not runtime ratios",
        "missing_metric_rule": "absent metrics are omitted from paired ratios; undefined ratios are null",
        "performance_improvement_required_for_pass": False,
    }


def build_comparison(
    *, baseline_path: Path = DEFAULT_BASELINE, current_path: Path = DEFAULT_CURRENT,
) -> dict:
    report = {
        "comparison": "aligned_hardware_frame_candidate_versions",
        "schema_version": 1,
        "sources": {"baseline": str(Path(baseline_path).resolve()), "current": str(Path(current_path).resolve())},
    }
    try:
        baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
        current = json.loads(Path(current_path).read_text(encoding="utf-8"))
        report.update(compare_reports(baseline, current))
    except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
        report.update({
            "status": "REFUSED", "reason": str(exc),
            "paired_comparisons": [], "summary_by_load_and_budget": [],
        })
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_comparison(baseline_path=args.baseline, current_path=args.current)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "paired_run_count": report.get("paired_run_count", 0), "reason": report.get("reason"), "output": str(args.output)}))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
