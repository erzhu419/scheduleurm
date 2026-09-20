"""Register and run the fixed 1% support-cap comparison on 20 new seeds."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path

from algorithm.experiments.aligned_frame_flow_decomposition import decompose_flow
from algorithm.experiments.aligned_hardware_frame_confirmation import _read, _write_new
from algorithm.experiments.aligned_hardware_frame_experiment import poisson_arrivals, simulate_frames
from algorithm.experiments.aligned_hardware_frame_mechanism import _assert_frozen_sources
from algorithm.experiments.aligned_hardware_frame_model import capacity_certificate, load_model
from algorithm.experiments.aligned_hardware_support_cap import POLICY_CAPS
from algorithm.experiments.aligned_hardware_support_cap_confirmation_stats import summarize_confirmation
from algorithm.experiments.aligned_hardware_waiting_cost import ARTIFACTS, PolicyRecorder, summarize_runs


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIRECTORY = ARTIFACTS / "aligned_hardware_support_cap_confirmation_20260910"
FROZEN_SOURCES = (
    "algorithm/budgeted_frame_candidates.py",
    "algorithm/waiting_cost_frame_candidates.py",
    "algorithm/experiments/aligned_hardware_frame_model.py",
    "algorithm/experiments/aligned_hardware_frame_experiment.py",
    "algorithm/experiments/aligned_frame_flow_decomposition.py",
    "algorithm/experiments/aligned_hardware_waiting_cost.py",
    "algorithm/experiments/aligned_hardware_support_cap.py",
    "algorithm/experiments/aligned_hardware_support_cap_confirmation_stats.py",
    "algorithm/experiments/aligned_hardware_support_cap_confirmation.py",
)
DEVELOPMENT = ARTIFACTS / "aligned_hardware_support_cap_development_20260910.json"


def _source_texts():
    return {name: (ROOT / name).read_text() for name in FROZEN_SOURCES}


def register(directory):
    _assert_frozen_sources()
    development = _read(DEVELOPMENT)
    previous = _read(ARTIFACTS / "aligned_hardware_frame_confirmation_v4_20260909/registration.json")
    used = set(previous["development_seeds"]) | set(previous["protocol"]["seeds"])
    used.update(development["protocol"]["base_protocol"]["seeds"])
    protocol = dict(development["protocol"]["base_protocol"], seeds=list(range(2041, 2061)))
    if used.intersection(protocol["seeds"]):
        raise ValueError("new confirmation seeds overlap previously used aligned-frame seeds")
    registration = {
        "registration": "fixed_1pct_support_cap_independent_seed_confirmation",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "previously_used_seeds": sorted(used), "protocol": protocol,
        "model": development["model"], "certificates": development["certificates"],
        "policy_max_relative_support_loss": POLICY_CAPS,
        "main_policy": "holding_cap_1pct", "main_budget": 26,
        "primary_loads": [0.8, 0.95], "family_alpha": 0.05, "family_size": 2,
        "primary_claim": "At each registered primary load, 1% cap K26 improves mean flow relative to v4 K26.",
        "estimator": "arithmetic mean of seed-paired mean-flow ratios; seed is the independent unit",
        "inference": "two Bonferroni-adjusted two-sided 97.5% Student t intervals; each upper endpoint must be below 1",
        "descriptive_controls": "load 0.50, the zero-cap rule, all other budgets, p90, flow components and timings",
        "frozen_selection_protocol": {key: development["protocol"][key] for key in (
            "eligibility", "cap_rationale", "objective", "arrival_rate_input", "drain_cost_interpretation", "candidate_generation")},
        "sample_size_rationale": "fixed 20 new seeds; interval precision is measured, with no guaranteed power for the small development effects",
        "stopping_rule": "run all 720 cases once; keep every registered seed; no cap changes, seed additions or outcome-dependent stopping",
        "execution_workers": 3,
        "execution_pass": "all jobs, budgets, support caps, queue recurrences, drift bounds and K120 support searches pass",
        "confirmation_pass": "both primary intervals are entirely below 1; separate from execution PASS",
        "scope": "independent arrivals at the two registered load points within the deterministic cached frame model",
    }
    directory.mkdir(parents=True, exist_ok=True)
    _write_new(directory / "source_snapshot.json", _source_texts())
    _write_new(directory / "registration.json", registration)
    return registration


def _run_seed(task):
    model, certificate, load, seed, protocol, policies = task
    horizon = protocol["arrival_horizon_s"]
    tape = poisson_arrivals(certificate["arrival_rates"], horizon, seed)
    rows = []
    for budget in protocol["budgets"]:
        for policy, cap in policies.items():
            recorder = PolicyRecorder(policy, sum(certificate["arrival_rates"]),
                                      max_relative_support_loss=1.0 if cap is None else cap)
            metrics = simulate_frames(model, tape, horizon, certificate, budget, selector=recorder)
            decomposition = decompose_flow(model, tape, recorder.frames, horizon)
            if abs(decomposition["mean_flow_s"] - metrics["mean_flow_s"]) > 1e-8:
                raise ValueError("flow decomposition disagrees with the registered simulator")
            rows.append({"load": load, "seed": seed, "budget": budget, "policy": policy,
                         "metrics": metrics, "decomposition": decomposition,
                         "selector_accounting": recorder.accounting()})
    return rows


def _reproduce_development(model, registration):
    development = _read(DEVELOPMENT)
    load, seed = 0.8, 41
    rows = _run_seed((model, capacity_certificate(model, load), load, seed,
                      registration["protocol"], registration["policy_max_relative_support_loss"]))
    expected = {(row["budget"], row["policy"]): row for row in development["runs"]
                if row["load"] == load and row["seed"] == seed}
    for row in json.loads(json.dumps(rows)):
        old = expected[(row["budget"], row["policy"])]
        for field in ("metrics", "decomposition"):
            if row[field] != old[field]:
                raise ValueError(f"development {field} changed for {(row['policy'], row['budget'])}")
        for key, value in row["selector_accounting"].items():
            if key not in ("selector_wall_s", "selector_mean_wall_ms") and value != old["selector_accounting"][key]:
                raise ValueError(f"development selector accounting changed: {key}")
    return {"status": "PASS", "load": load, "seed": seed, "runs": len(rows),
            "compared_fields": "all metrics, flow decompositions and deterministic selector accounting"}


def run_registered(directory):
    registration = _read(directory / "registration.json")
    snapshot = _read(directory / "source_snapshot.json")
    _assert_frozen_sources()
    if _source_texts() != snapshot:
        raise ValueError("registered sources changed before confirmation")
    if any((directory / name).exists() for name in ("runs.jsonl", "comparison.json", "confirmation.json")):
        raise ValueError("registered results already exist; this run will not overwrite them")
    model = load_model()
    if json.loads(json.dumps(model.snapshot())) != registration["model"]:
        raise ValueError("registered cached model changed; new seeds remain unused")
    protocol = registration["protocol"]
    certificates = [capacity_certificate(model, load) for load in protocol["loads"]]
    if json.loads(json.dumps(certificates)) != registration["certificates"]:
        raise ValueError("registered capacity certificates changed; new seeds remain unused")
    reproduction = _reproduce_development(model, registration)
    _write_new(directory / "development_reproduction.json", reproduction)
    print(json.dumps({"stage": "development_reproduced", **reproduction}), flush=True)
    tasks = [(model, certificate, load, seed, protocol, registration["policy_max_relative_support_loss"])
             for load, certificate in zip(protocol["loads"], certificates, strict=True) for seed in protocol["seeds"]]
    started = datetime.now(timezone.utc).isoformat()
    runs = []
    with (directory / "runs.jsonl").open("x") as stream:
        with ProcessPoolExecutor(max_workers=registration["execution_workers"]) as executor:
            for batch in executor.map(_run_seed, tasks):
                for row in batch:
                    stream.write(json.dumps(row) + "\n")
                stream.flush()
                runs.extend(batch)
                print(json.dumps({"load": batch[0]["load"], "seed": batch[0]["seed"], "runs_completed": len(runs)}), flush=True)
    if _source_texts() != snapshot:
        raise ValueError("registered sources changed during execution; confirmation withheld")
    caps = registration["policy_max_relative_support_loss"]
    capped = [row for row in runs if row["policy"] != "v4_support"]
    expected = {(load, seed, budget, policy) for load in protocol["loads"] for seed in protocol["seeds"]
                for budget in protocol["budgets"] for policy in caps}
    checks = {
        "registered_sources_unchanged": True,
        "complete_registered_roster": len(runs) == len(expected) and {(r["load"], r["seed"], r["budget"], r["policy"]) for r in runs} == expected,
        "all_jobs_complete": all(r["metrics"]["job_count"] == r["metrics"]["completed_jobs"] for r in runs),
        "construction_within_budget": all(r["metrics"]["maximum_candidate_attempts_per_frame"] <= r["budget"] for r in runs),
        "seed_support_preserved": all(r["selector_accounting"]["minimum_seed_support_margin"] >= 0 for r in capped),
        "support_floor_preserved": all(r["selector_accounting"]["minimum_support_floor_margin"] >= 0 for r in capped),
        "relative_loss_within_cap": all(r["selector_accounting"]["maximum_relative_support_loss"] <= caps[r["policy"]] + 1e-12 for r in capped),
        "holding_cost_no_worse_than_incumbent": all(r["selector_accounting"]["maximum_holding_cost_excess_over_support_incumbent"] <= 1e-10 for r in capped),
        "original_drift_bound_holds": all(r["metrics"]["maximum_expected_drift_bound_violation"] <= 1e-8 for r in runs),
        "queue_recursion_holds": all(r["metrics"]["maximum_queue_recursion_violation"] == 0 for r in runs),
        "k120_reference_exact": all(r["metrics"]["exact_selection_frame_count"] == r["metrics"]["frame_count"] for r in runs if r["budget"] == 120 and r["policy"] == "v4_support"),
        "k120_cap_support_search_exact": all(r["selector_accounting"]["support_search_exact_frames"] == r["metrics"]["frame_count"] for r in capped if r["budget"] == 120),
    }
    paired, summary = summarize_runs(runs, protocol["loads"], protocol["budgets"])
    for pair in paired:
        pair["paired_mean_flow_difference_s"] = pair["waiting_rule"]["mean_flow_s"] - pair["baseline"]["mean_flow_s"]
    comparison = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
                  "paired_comparisons": paired, "summary_by_load_and_budget": summary}
    _write_new(directory / "comparison.json", comparison)
    result = summarize_confirmation(comparison, registration) if comparison["status"] == "PASS" else {
        "status": "FAIL", "all_primary_claims_confirmed": False, "reason": "execution checks failed"}
    result.update({"run_count": len(runs), "checks": checks, "started_at_utc": started,
                   "finished_at_utc": datetime.now(timezone.utc).isoformat(), "development_reproduction": reproduction,
                   "sources": {name: str(directory / name) for name in ("registration.json", "runs.jsonl", "comparison.json")}})
    _write_new(directory / "confirmation.json", result)
    return result


def main():
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
                      "all_primary_claims_confirmed": result["all_primary_claims_confirmed"], "output": str(args.directory / "confirmation.json")}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
