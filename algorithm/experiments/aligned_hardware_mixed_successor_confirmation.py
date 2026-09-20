"""Confirm the frozen mixed-successor policy on new real and planning seeds."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

from algorithm.experiments.aligned_frame_flow_decomposition import decompose_flow
from algorithm.experiments.aligned_hardware_frame_confirmation import _write_new
from algorithm.experiments.aligned_hardware_frame_experiment import poisson_arrivals, simulate_frames
from algorithm.experiments.aligned_hardware_frame_model import capacity_certificate, load_model
from algorithm.experiments.aligned_hardware_lookahead import LookaheadRecorder, terminal_service_rates
from algorithm.experiments.aligned_hardware_mixed_successor_rollout import (
    CONTROLS, NEW_POLICY, MixedSuccessorRolloutRecorder,
)
from algorithm.experiments.aligned_hardware_poisson_rollout import PoissonRolloutRecorder
from algorithm.experiments.aligned_hardware_waiting_cost import ARTIFACTS, PolicyRecorder


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ARTIFACTS / "aligned_hardware_mixed_successor_confirmation_20260910.json"
DEVELOPMENT = ARTIFACTS / "aligned_hardware_mixed_successor_rollout_development_20260910.json"
REAL_SEEDS = tuple(range(4041, 4061))
PLANNING_SEEDS = (101, 102, 103)
BUDGET = 26
HORIZON = 20000.0
FROZEN_SNAPSHOTS = (
    ARTIFACTS / "aligned_hardware_lookahead_confirmation_20260910/source_snapshot.json",
    ARTIFACTS / "aligned_hardware_control_variate_development_20260910_fixed_sources.json",
    ARTIFACTS / "aligned_hardware_mixed_successor_continuation_development_20260910_fixed_sources.json",
)
NEW_SOURCE_FILES = (
    "algorithm/experiments/aligned_hardware_mixed_successor_confirmation.py",
    "algorithm/experiments/aligned_hardware_mixed_successor_confirmation_stats.py",
)


def _run_seed(task):
    """Use one real tape for two deterministic and six stochastic policy runs."""
    model, certificate, load, seed, protocol = task
    rates, mu = certificate["arrival_rates"], protocol["terminal_service_rates"]
    budget, horizon = protocol["budget"], protocol["arrival_horizon_s"]
    tape = poisson_arrivals(rates, horizon, seed)
    roster = ([(policy, None) for policy in CONTROLS[:2]]
              + [(policy, planning_seed) for policy in (CONTROLS[-1], NEW_POLICY)
                 for planning_seed in protocol["planning_seeds"]])
    rows = []
    for policy, planning_seed in roster:
        if policy == CONTROLS[0]:
            recorder = PolicyRecorder(policy, sum(rates), max_relative_support_loss=.01)
        elif policy == CONTROLS[1]:
            recorder = LookaheadRecorder(rates, mu, 1.)
        else:
            recorder_type = PoissonRolloutRecorder if policy == CONTROLS[-1] else MixedSuccessorRolloutRecorder
            recorder = recorder_type(rates, mu, planning_seed)
        metrics = simulate_frames(model, tape, horizon, certificate, budget, selector=recorder)
        decomposition = decompose_flow(model, tape, recorder.frames, horizon)
        if abs(decomposition["mean_flow_s"] - metrics["mean_flow_s"]) > 1e-8:
            raise ValueError("confirmation flow trace differs from the executed population")
        rows.append({"load": load, "seed": seed, "budget": budget, "policy": policy,
                     "planning_seed": planning_seed, "metrics": metrics,
                     "decomposition": decomposition,
                     "selector_accounting": recorder.accounting()})
    return rows


def check_confirmation_runs(runs, protocol, model):
    """Check complete populations and native budget/work identities for both rollouts."""
    expected = {
        (load, seed, protocol["budget"], policy, planning_seed)
        for load in protocol["loads"] for seed in protocol["seeds"]
        for policy in (*CONTROLS, NEW_POLICY)
        for planning_seed in (protocol["planning_seeds"]
                              if policy in (CONTROLS[-1], NEW_POLICY) else (None,))
    }
    actual = {(row["load"], row["seed"], row["budget"], row["policy"], row["planning_seed"])
              for row in runs}
    capped = [row for row in runs if row["policy"] != CONTROLS[0]]
    rollout = [row for row in runs if row["policy"] in (CONTROLS[-1], NEW_POLICY)]
    mixed = [row for row in runs if row["policy"] == NEW_POLICY]
    analytic = [row for row in runs if row["policy"] == CONTROLS[1]]
    counts = [row["selector_accounting"] for row in rollout]
    mixed_counts = [row["selector_accounting"] for row in mixed]
    populations = {}
    for row in runs:
        populations.setdefault((row["load"], row["seed"]), set()).add(row["metrics"]["job_count"])
    return {
        "complete_registered_roster": len(runs) == len(expected) and actual == expected,
        "all_jobs_complete": all(row["metrics"]["completed_jobs"] == row["metrics"]["job_count"] for row in runs),
        "same_population_within_each_real_tape": all(len(group) == 1 for group in populations.values()),
        "candidate_budget_holds": all(row["metrics"]["maximum_candidate_attempts_per_frame"] <= row["budget"] for row in runs),
        "flow_decomposition_holds": all(abs(row["decomposition"]["mean_flow_s"] - row["metrics"]["mean_flow_s"]) <= 1e-8 for row in runs),
        "seed_and_support_floors_hold": all(
            row["selector_accounting"]["minimum_seed_support_margin"] >= 0
            and row["selector_accounting"]["minimum_support_floor_margin"] >= 0
            and row["selector_accounting"]["maximum_relative_support_loss"] <= .01 + 1e-12
            for row in capped),
        "raw_and_mixed_stage_and_total_work_hold": all(
            count["maximum_stage_identity_error"] == count["maximum_budget_identity_error"] == 0
            for count in counts) and all(
                row["selector_accounting"]["root_attempts"] + row["selector_accounting"]["rollout_attempts"]
                == row["metrics"]["candidate_attempts"] for row in rollout),
        "raw_and_mixed_four_successors_per_comparison": all(
            count["rollout_attempts"] == 4 * count["rollout_frames"] for count in counts),
        "raw_and_mixed_population_and_draw_work_hold": all(
            count["maximum_path_population_error"] == count["maximum_random_draw_identity_error"] == 0
            and count["rollout_rng_initializations"] == len(model.workloads) for count in counts),
        "raw_and_mixed_selected_cost_no_worse_than_incumbent": all(
            count["maximum_rollout_cost_excess_over_incumbent"] <= 1e-8 for count in counts),
        "mixed_native_work_subtotals_hold": all(
            count["rollout_mixed_priority_coordinate_evaluations"] == count["rollout_class_evaluations"]
            and count["rollout_mixed_profile_probes"] == count["rollout_profile_probes"]
            and count["rollout_temporary_queue_updates"] <= count["rollout_queue_coordinate_evaluations"]
            for count in mixed_counts),
        "mixed_lane_work_bounded": all(
            count["rollout_temporary_queue_updates"] <= count["rollout_lane_fill_evaluations"]
            <= model.lanes * 4 * count["rollout_frames"] for count in mixed_counts),
        "analytic_cost_and_coordinate_accounting_hold": all(
            row["selector_accounting"]["maximum_objective_excess_over_support_incumbent"] <= 1e-8
            and row["selector_accounting"]["next_queue_coordinate_evaluations"]
            == row["selector_accounting"]["terminal_value_coordinate_evaluations"]
            == len(model.workloads) * row["selector_accounting"]["cost_candidate_evaluations"]
            for row in analytic),
        "queue_recursion_and_original_drift_hold": all(
            row["metrics"]["maximum_queue_recursion_violation"] == 0
            and row["metrics"]["maximum_expected_drift_bound_violation"] <= 1e-8 for row in runs),
    }


def _registered_sources():
    frozen = {}
    for path in FROZEN_SNAPSHOTS:
        frozen.update(json.loads(path.read_text()))
    if any((ROOT / name).read_text() != text for name, text in frozen.items()):
        raise ValueError("frozen raw or mixed source lineage changed before confirmation")
    return {**frozen, **{name: (ROOT / name).read_text() for name in NEW_SOURCE_FILES}}


def register(output=DEFAULT_OUTPUT):
    """Freeze source, model and protocol without generating any arrival tape."""
    from algorithm.experiments.aligned_hardware_mixed_successor_confirmation_stats import (
        ADOPTION_CRITERIA,
    )

    output = Path(output)
    protocol_path = output.with_name(output.stem + "_protocol.json")
    snapshot_path = output.with_name(output.stem + "_fixed_sources.json")
    runs_path = output.with_name(output.stem + "_runs.jsonl")
    if any(path.exists() for path in (output, protocol_path, snapshot_path, runs_path)):
        raise ValueError("confirmation artifacts already exist; frozen seeds will not be rerun or overwritten")
    previous = json.loads(DEVELOPMENT.read_text())
    if previous["status"] != "PASS":
        raise ValueError("mixed-successor development source did not pass")
    base = previous["protocol"]["base_protocol"]
    if base["arrival_horizon_s"] != HORIZON or BUDGET not in base["budgets"]:
        raise ValueError("confirmation must retain the development horizon and K26")
    model = load_model()
    certificates = {load: capacity_certificate(model, load) for load in base["loads"]}
    mu = terminal_service_rates(model, certificates[base["loads"][0]])
    if (json.loads(json.dumps(model.snapshot())) != previous["model"]
            or json.loads(json.dumps(list(certificates.values()))) != previous["certificates"]
            or list(mu) != previous["protocol"]["terminal_service_rates"]):
        raise ValueError("confirmation changed the frozen cached model, certificates or terminal rates")
    earlier = json.loads((ARTIFACTS / "aligned_hardware_lookahead_confirmation_20260910/registration.json").read_text())
    used_real_seeds = (set(earlier["previously_used_seeds"]) | set(earlier["protocol"]["seeds"])
                       | set(base["seeds"]))
    if used_real_seeds.intersection(REAL_SEEDS):
        raise ValueError("new confirmation real seeds overlap consumed development or confirmation seeds")
    if set(previous["protocol"]["planning_seeds"]).intersection(PLANNING_SEEDS):
        raise ValueError("new confirmation planning seeds overlap development planning seeds")
    source_texts = _registered_sources()
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "independent confirmation within the frozen cached frame model",
        "loads": base["loads"], "seeds": REAL_SEEDS, "planning_seeds": PLANNING_SEEDS,
        "budget": BUDGET, "arrival_horizon_s": HORIZON,
        "terminal_service_rates": mu, "policies": (*CONTROLS, NEW_POLICY),
        "model": model.snapshot(), "certificates": list(certificates.values()),
        "source_development": DEVELOPMENT.name,
        "previously_used_real_seeds": sorted(used_real_seeds),
        "roster": "three loads x twenty new real seeds x eight runs = 480 complete runs; per tape v4 once, analytic once, raw Poisson three and mixed successors three",
        "real_arrivals": "one tape per load/real seed, shared by all eight policies/repetitions, then drain its identical full population",
        "planning_arrivals": "frozen aligned-planning:{planning_seed}:{coordinate} streams; raw and mixed start from matched seeds, advance only on active calls and may diverge after changed decisions",
        "frozen_policy": "lane-by-lane mixed successors with q_i/mu_i priority and largest fitting pure profiles; original current candidate pool/support cap, four paid successors, shared Poisson horizon and original-mu fluid tail",
        "deterministic_controls": "v4 and analytic are evaluated once per real tape; their values are reused as matched denominators across planning repetitions",
        "adoption_criteria": ADOPTION_CRITERIA,
        "cost": "every row retains native candidate, bound, pricing, path, RNG and successor work; mixed lane/priority/profile/update subtotals and measured selector time are retained",
        "source_snapshot": "full text companion file written before any fresh tape is generated",
        "stopping_rule": "run all 480 registered cases once; retain all twenty real seeds and three planning repetitions; no tuning, additions or reruns after outcomes",
        "execution_workers": 3,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_new(protocol_path, protocol)
    _write_new(snapshot_path, source_texts)
    return protocol


def run_registered(output=DEFAULT_OUTPUT):
    """Validate the saved registration before starting the 480 fresh runs."""
    from algorithm.experiments.aligned_hardware_mixed_successor_confirmation_stats import (
        adoption_decision, summarize_confirmation,
    )

    output = Path(output)
    protocol_path = output.with_name(output.stem + "_protocol.json")
    snapshot_path = output.with_name(output.stem + "_fixed_sources.json")
    runs_path = output.with_name(output.stem + "_runs.jsonl")
    if not protocol_path.exists() or not snapshot_path.exists():
        raise ValueError("confirmation must be registered before running fresh seeds")
    if output.exists() or runs_path.exists():
        raise ValueError("confirmation results already exist; registered seeds will not be rerun or overwritten")
    protocol = json.loads(protocol_path.read_text())
    source_texts = json.loads(snapshot_path.read_text())
    if any((ROOT / name).read_text() != text for name, text in source_texts.items()):
        raise ValueError("registered sources changed; fresh confirmation seeds remain unused")
    model = load_model()
    if json.loads(json.dumps(model.snapshot())) != protocol["model"]:
        raise ValueError("registered model changed; fresh confirmation seeds remain unused")
    certificates = {load: capacity_certificate(model, load) for load in protocol["loads"]}
    if json.loads(json.dumps(list(certificates.values()))) != protocol["certificates"]:
        raise ValueError("registered certificates changed; fresh confirmation seeds remain unused")
    mu = terminal_service_rates(model, certificates[protocol["loads"][0]])
    if list(mu) != protocol["terminal_service_rates"]:
        raise ValueError("registered terminal rates changed; fresh confirmation seeds remain unused")
    tasks = [(model, certificate, load, seed, protocol)
             for load, certificate in certificates.items() for seed in protocol["seeds"]]
    started = perf_counter()
    runs = []
    with runs_path.open("x") as stream:
        with ProcessPoolExecutor(max_workers=protocol["execution_workers"]) as executor:
            for rows in executor.map(_run_seed, tasks):
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
                stream.flush()
                runs.extend(rows)
                print(json.dumps({"load": rows[0]["load"], "seed": rows[0]["seed"],
                                  "runs_completed": len(runs)}), flush=True)
    checks = check_confirmation_runs(runs, protocol, model)
    checks["registered_sources_unchanged_during_run"] = all(
        (ROOT / name).read_text() == text for name, text in source_texts.items())
    summary = summarize_confirmation(runs, protocol["loads"], protocol["seeds"], protocol["planning_seeds"])
    decision = adoption_decision(summary, checks)
    report = {
        "experiment": "aligned_hardware_mixed_successor_independent_confirmation_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "run_count": len(runs), "checks": checks, "protocol": protocol,
        "model": model.snapshot(), "certificates": list(certificates.values()),
        "runs": runs, "summary": summary, "adoption": decision,
        "wall_s": perf_counter() - started,
        "sources": {"protocol": str(protocol_path), "snapshot": str(snapshot_path),
                    "incremental_runs": str(runs_path)},
        "limitations": [
            "Independent units are real-arrival seed blocks; crossed load points and planning repetitions are retained within their seed blocks.",
            "Confirmation concerns the deterministic cached frame model and registered loads; it does not establish live hardware performance.",
            "Execution PASS and the registered statistical adoption decision are distinct outcomes.",
        ],
    }
    _write_new(output, report)
    return report


def build_confirmation(output=DEFAULT_OUTPUT):
    raise ValueError("use register(output) and then run_registered(output) as separate confirmation stages")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.register:
        result = register(args.output)
        print(json.dumps({"status": "REGISTERED", "seeds": result["seeds"],
                          "planning_seeds": result["planning_seeds"], "output": str(args.output)}))
        raise SystemExit(0)
    result = run_registered(args.output)
    print(json.dumps({"status": result["status"], "run_count": result["run_count"],
                      "adoption": result["adoption"], "output": str(args.output)}))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
