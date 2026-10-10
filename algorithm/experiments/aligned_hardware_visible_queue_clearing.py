"""Frozen performance test of commitment-aware visible-queue clearing.

The ordinary single-lane MaxWeight control and the new rule share the same
complete-job catalogue, causal arrival tape and asynchronous service clock.
Lane counts come from the measured catalogue metadata, not virtual R8.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from algorithm.experiments.aligned_hardware_frame_experiment import poisson_arrivals
from algorithm.experiments.aligned_hardware_frame_model import capacity_certificate
from algorithm.experiments.aligned_hardware_physical_catalogue_models import load_catalogues
from algorithm.experiments.aligned_hardware_performance_necessity import common_metrics, replay_matches
from algorithm.experiments.contextual_lane_refill_replay import simulate_contextual_trajectory
from algorithm.experiments.lane_refill_performance_replay import simulate_lane_refill_trajectory
from algorithm.lane_refill_simple_controls import select_single_lane_control
from algorithm.visible_queue_clearing_rollout import select_context_maxweight, select_visible_queue_clearing


ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "md/experiment_artifacts/aligned_hardware_visible_queue_clearing_development_20260930"
LABELS = ("maxweight", "clearing_rollout")
SOURCES = (
    "algorithm/visible_queue_clearing_rollout.py", "algorithm/lane_refill_simple_controls.py",
    "algorithm/experiments/aligned_hardware_visible_queue_clearing.py",
    "algorithm/experiments/contextual_lane_refill_replay.py",
    "algorithm/experiments/visible_queue_clearing_summary.py",
    "algorithm/experiments/aligned_hardware_performance_necessity.py",
    "algorithm/experiments/lane_refill_performance_replay.py",
    "algorithm/experiments/integer_capacity_trajectory_replay.py",
    "algorithm/experiments/aligned_hardware_shared_threshold_scaling_screen.py",
    "algorithm/experiments/aligned_hardware_frame_experiment.py",
    "algorithm/experiments/aligned_hardware_frame_model.py",
    "algorithm/experiments/aligned_hardware_physical_catalogue_models.py",
    "algorithm/experiments/unified_hardware_or_replay.py",
)


def build_protocol(models):
    loads, seeds, horizon, end = (.80, .95), (41, 42, 43), 600.0, 1200.0
    tapes, capacities = [], []
    for model in models:
        for coordinate in range(len(model.workloads)):
            if not any(not option.loaded and sum(option.requirements) == 1 and option.requirements[coordinate] == 1
                       for option in model.options):
                raise ValueError("each nonempty class needs a dedicated singleton to bound clearing work")
        for load in loads:
            certificate = capacity_certificate(model, load)
            capacities.append({"node": model.node, "lanes": model.lanes, "load": load,
                               **{key: certificate[key] for key in ("arrival_rates", "full_boundary_scale", "capacity_basis")}})
            for seed in seeds:
                tapes.append({"tape_id": f"{model.node}:R{model.lanes}:load{load}:seed{seed}",
                              "node": model.node, "lanes": model.lanes, "load": load, "seed": seed,
                              "arrivals": poisson_arrivals(certificate["arrival_rates"], horizon, seed)})
    return json.loads(json.dumps({
        "experiment": "visible_queue_clearing_development_v1", "models": [model.snapshot() for model in models],
        "loads": loads, "arrival_seeds": seeds, "arrival_horizon": horizon, "observation_end": end,
        "tapes": tapes, "capacities": capacities, "labels": LABELS,
        "execution_order": [(model.node, model.lanes) for model in models],
        "performance": {"min_gain": .01, "group_secondary_tolerance": .05},
        "method": "At each first free lane, enumerate every currently feasible real root option only if0<N=sum(Q)<=R*B, B=max total local batch width. For each root, clear all current visible waiting jobs with ordinary exact single-lane MaxWeight continuation and zero hypothetical new arrivals. Preserve all other lane commitments and all real target/resident completion offsets. Minimize total predicted completion time relative to now; keep the MaxWeight root on ties, and change it only for strict exact-rational improvement. Outside this model-derived finite queue envelope keep ordinary MaxWeight.",
        "budget": "At most m real roots and m*(N-1) continuation calls when eligible; each continuation scans m original options. Root, preprocessing and continuation option probes are charged separately. No LP/IP or future arrival reads. Bounded small-queue rollout is not an enumeration of the global action family.",
        "clock": "Single-lane ascending-index dispatch at min full lane-availability time; same1s idle polling. Arrivals never trigger dispatch. Physical service only, no computing delay. Loaded targets completing early cannot free resident commitments. Stop new calls atT and retain scheduled completion and queued tails.",
        "primary": "Mean of the24 paired-tape T all-released-job restricted mean flows improves at least1% versus ordinary MaxWeight; each load mean does not worsen. All16 node/load/H,T groups retain common-ID mean/P90 and unfinished counts; P90 and unfinished allow at most5% group regression, zero unfinished control requires zero. Pooled H/T unfinished counts cannot increase. Preserve every seed/group regression.",
        "statistics": "Development screen only. Same seed shares underlying random streams across catalogue/load cells; aggregate each of3 independent seeds over8 cells, do not treat24 tapes as24 independent samples. No post-readout parameter changes; an advance needs separately frozen new-seed confirmation.",
        "validation": "The new contextual engine's ordinary MaxWeight trace must reproduce the existing serial engine action/start/Q/lane/service/duration and every job timestamp on every tape. Check bounded rollout work, queue-envelope fallback and predicted nonincrease. A valid support gap is recorded; the new policy need not maximize MaxWeight support.",
        "limitations": "Complete-job deterministic cached independent-lane model using catalogue-reported2/4 lanes; not physical execution. Predicting no new arrivals is a queue-clearing proxy, not an online flow oracle. Original synchronous drift proofs do not certify this asynchronous policy. Development seeds are reused; wall time is not a success metric.",
        "execution_sources": {path: (ROOT / path).read_text() for path in SOURCES},
    }))


def rollout_checks(result, model):
    m, limit = len(model.options), model.lanes * max(sum(option.requirements) for option in model.options)
    for frame in result["frames"]:
        decision, n = frame["rollout"], sum(frame["queue"])
        if decision["queue_limit"] != limit or decision["eligible"] != (0 < n <= limit):
            return False
        if decision["candidate_count"] > m or decision["rollout_calls"] > decision["candidate_count"] * max(0, n - 1):
            return False
        if decision["rollout_option_probes"] != m * decision["rollout_calls"]:
            return False
        if decision["root_option_probes"] != (3 * m if decision["eligible"] else m):
            return False
        if decision["eligible"]:
            if decision["chosen_predicted_completion_s"] > decision["baseline_predicted_completion_s"]:
                return False
        elif decision["changed"] or decision["rollout_calls"] or decision["candidate_count"]:
            return False
    return True


def run_tape(model, tape, protocol):
    policies = {label: simulate_contextual_trajectory(model, tape["arrivals"], protocol["arrival_horizon"],
                    protocol["observation_end"], selector, policy=label)
                for label, selector in zip(LABELS, (select_context_maxweight, select_visible_queue_clearing), strict=True)}
    shadow, _ = simulate_lane_refill_trajectory(model, tape["arrivals"], protocol["arrival_horizon"],
                    protocol["observation_end"], 0, select_single_lane_control, policy="maxweight", dispatch_rule="single")
    identical = replay_matches(policies["maxweight"], json.loads(json.dumps(shadow)))
    checks = rollout_checks(policies["clearing_rollout"], model)
    row = {**{key: tape[key] for key in ("tape_id", "node", "lanes", "load", "seed")},
           "baseline_reproduction_passed": identical, "rollout_checks_passed": checks,
           "snapshots": {cutoff: common_metrics(policies, protocol[cutoff]) for cutoff in ("arrival_horizon", "observation_end")},
           "work": {label: result["rollout_work"] for label, result in policies.items()}}
    return row, policies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    models, base = load_catalogues(), args.output
    protocol_path = base.with_name(base.name + "_protocol.json")
    if args.register:
        payload = {**build_protocol(models), "registered_at_utc": datetime.now(timezone.utc).isoformat()}
        with protocol_path.open("x") as out:
            json.dump(payload, out, separators=(",", ":"), allow_nan=False)
        print(protocol_path, flush=True)
        return
    protocol = json.loads(protocol_path.read_text())
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("registered models, physical-lane tapes, method or performance criteria changed")
    output, trajectories = base.with_suffix(".json"), base.with_name(base.name + "_trajectories.json.gz")
    if output.exists() or trajectories.exists():
        raise FileExistsError("preserve existing registered results")
    rows, runs = [], []
    for tape in protocol["tapes"]:
        row, policies = run_tape(next(model for model in models if model.node == tape["node"]), tape, protocol)
        rows.append(row)
        runs.append({"tape_id": tape["tape_id"], "policies": policies})
        print(f"{tape['tape_id']}: paired performance done; serial identity={row['baseline_reproduction_passed']} rollout valid={row['rollout_checks_passed']}", flush=True)
    with gzip.open(trajectories, "xt") as out:
        json.dump({"registered_at_utc": protocol["registered_at_utc"], "runs": runs}, out, separators=(",", ":"), allow_nan=False)
    from algorithm.experiments.visible_queue_clearing_summary import summarize
    result = summarize(rows, protocol)
    with output.open("x") as out:
        json.dump(result, out, separators=(",", ":"), allow_nan=False)
    print(json.dumps({key: result[key] for key in ("decision", "execution_checks_passed", "performance_gain_established", "pooled")}), flush=True)


if __name__ == "__main__":
    main()
