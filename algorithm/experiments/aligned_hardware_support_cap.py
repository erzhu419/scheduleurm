"""Development comparison of fixed 1% and zero paid-support loss caps.

Keep the original arrival roster, holding objective and seed floor. The fixed
1% cap is the main design; zero loss distinguishes support ties from sacrifices.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean

from algorithm.experiments.aligned_hardware_waiting_cost import (
    ARTIFACTS, PolicyRecorder, summarize_runs,
)
from algorithm.experiments.aligned_frame_flow_decomposition import decompose_flow
from algorithm.experiments.aligned_hardware_frame_experiment import poisson_arrivals, simulate_frames
from algorithm.experiments.aligned_hardware_frame_mechanism import _assert_frozen_sources
from algorithm.experiments.aligned_hardware_frame_model import capacity_certificate, load_model


DEFAULT_OUTPUT = ARTIFACTS / "aligned_hardware_support_cap_development_20260910.json"
POLICY_CAPS = {"v4_support": None, "holding_cap_zero": 0.0, "holding_cap_1pct": 0.01}


def build_experiment(output=DEFAULT_OUTPUT):
    _assert_frozen_sources()
    development = json.loads((ARTIFACTS / "aligned_hardware_frame_experiment_v4_20260909.json").read_text())
    previous = json.loads((ARTIFACTS / "aligned_hardware_waiting_cost_development_20260909.json").read_text())
    base = development["protocol"]
    model = load_model()
    if not (json.loads(json.dumps(model.snapshot())) == development["model"] == previous["model"]):
        raise ValueError("support-cap comparison changed the model")
    if previous["protocol"]["base_protocol"] != base:
        raise ValueError("prior seed-only comparison has a different development roster")
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "development comparison; no independent confirmation",
        "base_protocol": base, "policy_max_relative_support_loss": POLICY_CAPS,
        "main_policy": "holding_cap_1pct", "main_budget": 26,
        "eligibility": "S(a) >= max(S_seed, (1-r)*S_inc); S_inc is the same paid pool's best support",
        "cap_rationale": "1% is a fixed conservative design choice, not a fitted or optimal parameter",
        "zero_cap_control": "preserves paid incumbent support while allowing holding-cost changes among ties",
        "objective": previous["protocol"]["new_objective"],
        "arrival_rate_input": previous["protocol"]["arrival_rate_input"],
        "drain_cost_interpretation": previous["protocol"]["drain_cost_interpretation"],
        "candidate_generation": previous["protocol"]["candidate_generation"],
        "reference": "frozen v4 K120; both cap policies retain their own selected support gap",
        "prior_seed_only_comparison": "reuse the previous 36 seed-only runs; do not rerun or overwrite them",
        "stopping_rule": "run all 108 cases once; no cap relaxation, grid search or budget selection after results",
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_name(output.stem + "_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    loads, seeds, budgets = (base[key] for key in ("loads", "seeds", "budgets"))
    horizon = base["arrival_horizon_s"]
    original = {(row["load"], row["seed"], row["budget"]): row for row in development["runs"]}
    runs, certificates = [], []
    for load in loads:
        certificate = capacity_certificate(model, load)
        certificates.append(certificate)
        for seed in seeds:
            tape = poisson_arrivals(certificate["arrival_rates"], horizon, seed)
            for budget in budgets:
                for policy, cap in POLICY_CAPS.items():
                    recorder = PolicyRecorder(policy, sum(certificate["arrival_rates"]),
                                              max_relative_support_loss=1.0 if cap is None else cap)
                    metrics = simulate_frames(model, tape, horizon, certificate, budget, selector=recorder)
                    decomposition = decompose_flow(model, tape, recorder.frames, horizon)
                    if abs(decomposition["mean_flow_s"] - metrics["mean_flow_s"]) > 1e-8:
                        raise ValueError("flow decomposition disagrees with the frozen simulator")
                    if policy == "v4_support":
                        observed = json.loads(json.dumps(metrics))
                        changed = [key for key, value in observed.items() if original[(load, seed, budget)][key] != value]
                        if changed:
                            raise ValueError(f"v4 baseline changed during support-cap test: {changed}")
                    runs.append({"load": load, "seed": seed, "budget": budget, "policy": policy,
                                 "metrics": metrics, "decomposition": decomposition,
                                 "selector_accounting": recorder.accounting()})
            print(json.dumps({"load": load, "seed": seed, "runs_completed": len(runs)}), flush=True)
    _assert_frozen_sources()
    paired, summary = summarize_runs(runs, loads, budgets)
    prior = {(row["load"], row["seed"], row["budget"]): row for row in previous["runs"]
             if row["policy"] == "seed_guarded_holding_v1"}
    for row in paired:
        old = prior[(row["load"], row["seed"], row["budget"])]["metrics"]["mean_flow_s"]
        row["prior_seed_only_mean_flow_s"] = old
        row["flow_ratio_over_prior_seed_only"] = row["waiting_rule"]["mean_flow_s"] / old
    for group in summary:
        group["mean_paired_flow_ratio_over_prior_seed_only"] = mean(
            row["flow_ratio_over_prior_seed_only"] for row in paired
            if (row["load"], row["budget"], row["policy"]) == (group["load"], group["budget"], group["policy"])
        )
    capped = [row for row in runs if row["policy"] != "v4_support"]
    checks = {
        "frozen_sources_and_v4_runs_unchanged": True,
        "all_jobs_complete": all(row["metrics"]["completed_jobs"] == row["metrics"]["job_count"] for row in runs),
        "construction_within_budget": all(row["metrics"]["maximum_candidate_attempts_per_frame"] <= row["budget"] for row in runs),
        "seed_support_preserved": all(row["selector_accounting"]["minimum_seed_support_margin"] >= 0 for row in capped),
        "support_floor_preserved": all(row["selector_accounting"]["minimum_support_floor_margin"] >= 0 for row in capped),
        "relative_loss_within_registered_cap": all(row["selector_accounting"]["maximum_relative_support_loss"] <= POLICY_CAPS[row["policy"]] + 1e-12 for row in capped),
        "holding_cost_no_worse_than_incumbent": all(row["selector_accounting"]["maximum_holding_cost_excess_over_support_incumbent"] <= 1e-10 for row in capped),
        "original_drift_bound_holds": all(row["metrics"]["maximum_expected_drift_bound_violation"] <= 1e-8 for row in runs),
        "queue_recursion_holds": all(row["metrics"]["maximum_queue_recursion_violation"] == 0 for row in runs),
        "k120_support_search_exact": all(row["selector_accounting"]["support_search_exact_frames"] == row["metrics"]["frame_count"] for row in capped if row["budget"] == 120),
    }
    report = {
        "experiment": "aligned_hardware_support_cap_development_v1",
        "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "protocol": protocol, "model": model.snapshot(), "certificates": certificates,
        "runs": runs, "paired_comparisons": paired, "summary_by_load_and_budget": summary,
        "limitations": [
            "The relative cap concerns only the paid support incumbent, not full-space support or mean flow.",
            "The selected certificate satisfies U-S_selected <= (U-S_inc)+r*S_inc; the seed guarantee still holds.",
            "The support floor can increase with candidate budget, so holding cost need not decrease with budget.",
            "The holding objective is myopic and optimized only on the paid pool; Lambda remains a stationary proxy during drain.",
            "Execution PASS does not mean performance improvement; results use reused development seeds and deterministic cached durations.",
            "Wall time is descriptive local timing; extra cost evaluations and offset reads are reported separately from candidate attempts.",
        ],
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build_experiment(args.output)
    print(json.dumps({"status": result["status"], "runs": len(result["runs"]), "output": str(args.output)}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
