"""Register and run one frozen v4 confirmation on ten unused arrival seeds.

Run ``--register`` before ``--run``. Both commands use the fixed protocol below;
the run never changes its seed roster or selects a better-performing variant.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from algorithm.budgeted_frame_candidates import select_budgeted_frame as select_v4
from algorithm.experiments.aligned_hardware_frame_comparison import compare_reports
from algorithm.experiments.aligned_hardware_frame_confirmation_stats import summarize_confirmation
from algorithm.experiments.aligned_hardware_frame_experiment import build_experiment
from algorithm.experiments.aligned_hardware_frame_model import capacity_certificate, load_model
from algorithm.experiments.baselines.aligned_frame_candidates_v1 import select_budgeted_frame as select_v1


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md/experiment_artifacts"
DEFAULT_DIRECTORY = ARTIFACT_ROOT / "aligned_hardware_frame_confirmation_v4_20260909"
FROZEN_SOURCES = (
    "algorithm/budgeted_frame_candidates.py",
    "algorithm/experiments/baselines/aligned_frame_candidates_v1.py",
    "algorithm/experiments/aligned_hardware_frame_model.py",
    "algorithm/experiments/aligned_hardware_frame_experiment.py",
    "algorithm/experiments/aligned_hardware_frame_comparison.py",
    "algorithm/experiments/aligned_hardware_frame_confirmation_stats.py",
    "algorithm/experiments/aligned_hardware_frame_confirmation.py",
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_new(path: Path, data: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


def _source_texts() -> dict[str, str]:
    return {name: (REPO_ROOT / name).read_text(encoding="utf-8") for name in FROZEN_SOURCES}


def register(directory: Path) -> dict:
    development = _read(ARTIFACT_ROOT / "aligned_hardware_frame_experiment_v4_20260909.json")
    used_seeds = set()
    for suffix in ("", "_v2", "_v3", "_v4"):
        report = _read(ARTIFACT_ROOT / f"aligned_hardware_frame_experiment{suffix}_20260909.json")
        used_seeds.update(report["protocol"]["seeds"])
    protocol = dict(development["protocol"], seeds=list(range(1041, 1051)))
    if used_seeds.intersection(protocol["seeds"]):
        raise ValueError("confirmation seeds overlap the development roster")
    registration = {
        "registration": "aligned_frame_v4_independent_arrival_seed_confirmation",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_seeds": sorted(used_seeds),
        "protocol": protocol,
        "model": development["model"], "certificates": development["certificates"],
        "selectors": {"v1": "historical_weight_order_uncapped_bounds", "v4": development["selector"]},
        "main_budget": 26, "main_load": 0.50,
        "equivalence_margin": 0.002, "family_alpha": 0.05, "family_size": 4,
        "main_claims": [
            "At load 0.50, v4 K26 improves mean flow relative to v1 K26.",
            "At each of the three loads, v4 K26 mean flow is within 0.2% of v4 exact.",
        ],
        "estimator": "arithmetic mean of per-seed paired mean-flow ratios",
        "inference": "four Bonferroni-adjusted two-sided Student t intervals; independent unit is the arrival seed",
        "stopping_rule": "run all ten registered seeds once; retain all results; no tuning or roster extension",
        "execution_pass": "all jobs complete; registered model, budget, queue recurrence and exact reference checks pass",
        "confirmation_pass": "each primary confidence interval satisfies its registered claim; separate from execution",
        "scope": "independent arrival realizations within the same deterministic hardware-cache frame model",
    }
    directory.mkdir(parents=True, exist_ok=True)
    _write_new(directory / "source_snapshot.json", _source_texts())
    _write_new(directory / "registration.json", registration)
    return registration


def _validate_historical_baseline() -> dict:
    """Replay one already-used development seed before accessing any holdout."""
    development = _read(ARTIFACT_ROOT / "aligned_hardware_frame_experiment_20260909.json")
    protocol = development["protocol"]
    reproduced = build_experiment(
        horizon=protocol["arrival_horizon_s"], loads=(0.50,), seeds=(41,),
        budgets=protocol["budgets"], selector=select_v1,
    )
    # Normalize tuples exactly as the original JSON writer did.
    observed = json.loads(json.dumps(reproduced))["runs"]
    expected = [row for row in development["runs"] if row["load"] == 0.50 and row["seed"] == 41]
    for old, new in zip(expected, observed, strict=True):
        changed = [key for key, value in old.items() if new[key] != value]
        if changed:
            raise ValueError(f"historical v1 reproduction changed {changed} at K={old['budget']}")
    return {"status": "PASS", "load": 0.50, "seed": 41, "budgets": protocol["budgets"],
            "compared_fields": "all previously recorded run fields; new threshold instrumentation excluded"}


def run_registered(directory: Path) -> dict:
    registration = _read(directory / "registration.json")
    source_snapshot = _read(directory / "source_snapshot.json")
    if _source_texts() != source_snapshot:
        raise ValueError("registered source changed; this confirmation cannot run under the frozen protocol")
    output_paths = [directory / name for name in ("v1.json", "v4.json", "comparison.json", "confirmation.json")]
    if any(path.exists() for path in output_paths):
        raise ValueError("confirmation results already exist; the registered run will not overwrite them")
    # Cache data can change without changing Python source. Reject that before
    # generating any registered arrival tape, rather than after using holdout.
    model = load_model()
    if json.loads(json.dumps(model.snapshot())) != registration["model"]:
        raise ValueError("current cached model differs from registration; holdout seeds remain unused")
    certificates = [capacity_certificate(model, load) for load in registration["protocol"]["loads"]]
    if json.loads(json.dumps(certificates)) != registration["certificates"]:
        raise ValueError("current capacity certificates differ from registration; holdout seeds remain unused")
    baseline_check = _validate_historical_baseline()
    _write_new(directory / "historical_baseline_check.json", baseline_check)
    print(json.dumps({"stage": "historical_baseline_reproduced", **baseline_check}), flush=True)
    protocol = registration["protocol"]
    started = datetime.now(timezone.utc).isoformat()
    reports = {}
    for name, selector in (("v1", select_v1), ("v4", select_v4)):
        report = build_experiment(
            horizon=protocol["arrival_horizon_s"], loads=protocol["loads"],
            seeds=protocol["seeds"], budgets=protocol["budgets"], selector=selector,
        )
        for field in ("model", "protocol", "certificates"):
            if json.loads(json.dumps(report[field])) != registration[field]:
                raise ValueError(f"{name} {field} differs from the registration")
        report.update({
            "experiment": f"aligned_hardware_frame_{name}_confirmation",
            "selector": registration["selectors"][name],
            "evaluation_role": "registered independent arrival-seed confirmation; no further tuning",
        })
        reports[name] = report
        _write_new(directory / f"{name}.json", report)
        print(json.dumps({"stage": f"{name}_finished", "status": report["status"], "runs": len(report["runs"])}), flush=True)
    if _source_texts() != source_snapshot:
        raise ValueError("registered source changed during execution; confirmation withheld")
    if not all(report["status"] == "PASS" for report in reports.values()):
        result = {"status": "FAIL", "reason": "execution checks failed; no primary claim confirmed"}
    else:
        comparison = compare_reports(reports["v1"], reports["v4"])
        _write_new(directory / "comparison.json", comparison)
        result = summarize_confirmation(comparison, registration)
    result.update({
        "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "registered_source_unchanged": True,
        "historical_baseline_reproduction": baseline_check,
        "run_count": sum(len(report["runs"]) for report in reports.values()),
        "sources": {"registration": str(directory / "registration.json"),
                    "baseline": str(directory / "v1.json"), "current": str(directory / "v4.json"),
                    "paired_comparison": str(directory / "comparison.json")},
    })
    _write_new(directory / "confirmation.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    args = parser.parse_args()
    if args.register:
        result = register(args.directory)
        print(json.dumps({"status": "REGISTERED", "seeds": result["protocol"]["seeds"], "directory": str(args.directory)}))
        return 0
    result = run_registered(args.directory)
    print(json.dumps({"status": result["status"], "run_count": result["run_count"],
                      "all_primary_claims_confirmed": result.get("all_primary_claims_confirmed"),
                      "output": str(args.directory / "confirmation.json")}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
