"""Compare exact optimized/native proposals on four frozen development states."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter, process_time

from algorithm.experiments.aligned_hardware_performance_necessity import load_models
from algorithm.exact_reuse_lane_rollout import ExactReuseTwoStepCurrentPolicyBGIncumbentLaneRollout
from algorithm.two_step_current_policy_bg_incumbent_lane_rollout import TwoStepCurrentPolicyBGIncumbentLaneRollout


ROOT = Path(__file__).resolve().parents[2]
STEM = "aligned_hardware_exact_reuse_benchmark_20261001"
DIRECTORY = ROOT / "md/experiment_artifacts"
NEW_SOURCES = (
    "algorithm/exact_reuse_lane_rollout.py", "algorithm/exact_reuse_inner_decision.py",
    "algorithm/dyadic_background_rollout.py",
    "algorithm/experiments/aligned_hardware_exact_reuse_benchmark.py",
)


def recording_policy(factory, model, tape, samples, cap):
    class Recording(factory):
        def score_background_roots(self, *context):
            result = super().score_background_roots(*context)
            self.last_scores = result
            return result
    planner = Recording(model.options, f"arrival-plan-v1:{model.node}:{tape['load']}:online", samples, cap)
    planner.last_scores = None
    return planner


def register(directory):
    from algorithm.experiments import aligned_hardware_two_step_current_policy_root_ranking as ranking
    from algorithm.experiments.two_step_bg_incumbent_continuation_cross_replay import reconstruct_prefix_states
    frozen = json.loads(ranking.artifact(ranking.BASE, "_protocol.json").read_text())
    for name, source in frozen["execution_sources"].items():
        if (ROOT / name).read_text() != source:
            raise ValueError("frozen source changed: " + name)
    factual = ranking.saved_runs(frozen)
    contexts = []
    for model in load_models(frozen["models"]):
        anchors = [anchor for anchor in frozen["anchors"]
                   if next(tape["node"] for tape in frozen["tapes"]
                           if tape["tape_id"] == anchor["tape_id"]) == model.node]
        anchor = min(anchors, key=lambda row: row["visible_cohort_jobs"])
        tape = next(tape for tape in frozen["tapes"] if tape["tape_id"] == anchor["tape_id"])
        planner = recording_policy(TwoStepCurrentPolicyBGIncumbentLaneRollout, model, tape,
                                   frozen["planning_samples"], frozen["planning_event_cap"])
        states = reconstruct_prefix_states(planner, factual[tape["tape_id"]],
                                           [anchor["frame_index"]], Counter())
        frame = factual[tape["tape_id"]]["frames"][anchor["frame_index"]]
        contexts.append({"anchor": anchor, "tape": {key: tape[key] for key in
            ("tape_id", "node", "lanes", "load", "seed")},
            "state": states[anchor["frame_index"]],
            "context": {key: frame[key] for key in (
                "queue", "available_before", "start", "lane_ids", "waiting_ages", "observed_arrival_counts")}})
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "exact_computation_reuse_four_native_proposals_v1",
        "source_ranking_registration": frozen["registered_at_utc"],
        "selection": "Smallest visible-cohort anchor for each of the four physical models in the frozen22 states; roster-order ties. Bound diagnostic cost without changing8/cap128. No tuning or service-performance gate.",
        "scope": "Four paired complete native proposals at8/cap128, exact scores/decision/all three controller states. Timings exclude state reconstruction and constructors. No new full service trajectory; speedup applies only to these states.",
        "models": frozen["models"], "planning_samples": frozen["planning_samples"],
        "planning_event_cap": frozen["planning_event_cap"], "contexts": contexts,
        "execution_sources": {**frozen["execution_sources"],
                              **{name: (ROOT / name).read_text() for name in NEW_SOURCES}},
    }
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / (STEM + "_protocol.json")).open("x") as out:
        json.dump(protocol, out, separators=(",", ":"), allow_nan=False)


def rng_tuple(value):
    return value[0], tuple(value[1]), value[2]


def planner_states(planner):
    states = []
    while planner is not None:
        states.append({"rng": planner.rng.getstate(), "counter": planner.decision_counter,
                       "rates": planner.arrival_rates,
                       "counts": getattr(planner, "_outer_observed_counts", None)})
        planner = getattr(planner, "inner_planner", None)
    return states


def load_protocol(directory):
    protocol = json.loads((directory / (STEM + "_protocol.json")).read_text())
    for name, source in protocol["execution_sources"].items():
        if (ROOT / name).read_text() != source:
            raise ValueError("registered source changed: " + name)
    return protocol


def run(index, directory):
    protocol = load_protocol(directory)
    row = protocol["contexts"][index]
    tape, context = row["tape"], row["context"]
    model = next(model for model in load_models(protocol["models"]) if model.node == tape["node"])
    outputs, timing = {}, {}
    for label, factory in (("native", TwoStepCurrentPolicyBGIncumbentLaneRollout),
                           ("optimized", ExactReuseTwoStepCurrentPolicyBGIncumbentLaneRollout)):
        planner = recording_policy(factory, model, tape,
            protocol["planning_samples"], protocol["planning_event_cap"])
        planner.rng.setstate(rng_tuple(row["state"]["rng_state"]))
        planner.decision_counter = row["state"]["decision_counter"]
        wall, cpu = perf_counter(), process_time()
        decision = planner(model.options, tuple(context["queue"]), tuple(context["available_before"]),
            context["start"], context["lane_ids"][0],
            tuple(tuple(group) for group in context["waiting_ages"]),
            tuple(context["observed_arrival_counts"]))
        timing[label] = {"wall_s": perf_counter() - wall, "cpu_s": process_time() - cpu}
        outputs[label] = (asdict(decision), planner.last_scores, planner_states(planner))
        if label == "optimized":
            timing[label]["executed_work"] = planner.reuse_work
        print(tape["tape_id"], label, timing[label]["wall_s"], flush=True)
    checks = {"decision_and_logical_work_exact": outputs["native"][0] == outputs["optimized"][0],
              "fraction_scores_exact": outputs["native"][1] == outputs["optimized"][1],
              "three_controller_states_exact": outputs["native"][2] == outputs["optimized"][2]}
    result = {"registered_at_utc": protocol["registered_at_utc"], "anchor_index": index,
        "tape_id": tape["tape_id"], "queue": context["queue"], "checks": checks,
        "timing": timing, "cpu_speedup": timing["native"]["cpu_s"] / timing["optimized"]["cpu_s"],
        "wall_speedup": timing["native"]["wall_s"] / timing["optimized"]["wall_s"],
        "native_option_index": outputs["native"][0]["selection"]["option_indices"][0],
        "optimized_option_index": outputs["optimized"][0]["selection"]["option_indices"][0]}
    with (directory / f"{STEM}_{index:02d}.json").open("x") as out:
        json.dump(result, out, separators=(",", ":"), allow_nan=False)
    if not all(checks.values()):
        raise ValueError("exact optimized/native proposal mismatch: " + tape["tape_id"])


def merge(directory):
    protocol = load_protocol(directory)
    records = [json.loads((directory / f"{STEM}_{index:02d}.json").read_text())
               for index in range(len(protocol["contexts"]))]
    if any(row["anchor_index"] != index or row["registered_at_utc"] != protocol["registered_at_utc"]
           or row["tape_id"] != protocol["contexts"][index]["tape"]["tape_id"]
           for index, row in enumerate(records)):
        raise ValueError("all four matching paired proposals required")
    passed = all(all(row["checks"].values()) for row in records)
    totals = {label: {key: sum(row["timing"][label][key] for row in records)
                      for key in ("wall_s", "cpu_s")} for label in ("native", "optimized")}
    result = {"registered_at_utc": protocol["registered_at_utc"], "exact_match_passed": passed,
        "decision": "EXACT_REUSE_BENCHMARK_PASS" if passed else "EXACT_REUSE_BENCHMARK_FAIL",
        "records": records, "totals": totals,
        "pooled_cpu_speedup": totals["native"]["cpu_s"] / totals["optimized"]["cpu_s"],
        "new_full_trajectories": 0, "scope": protocol["scope"]}
    with (directory / (STEM + ".json")).open("x") as out:
        json.dump(result, out, separators=(",", ":"), allow_nan=False)
    print(json.dumps({key: result[key] for key in ("decision", "pooled_cpu_speedup")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run", type=int, choices=range(4))
    mode.add_argument("--merge", action="store_true")
    parser.add_argument("--artifact-dir", type=Path, default=DIRECTORY)
    args = parser.parse_args()
    if args.register:
        register(args.artifact_dir)
    elif args.run is not None:
        run(args.run, args.artifact_dir)
    else:
        merge(args.artifact_dir)


if __name__ == "__main__":
    main()
