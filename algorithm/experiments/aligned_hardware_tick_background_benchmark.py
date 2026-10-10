"""Paired exact execution/cost checks of the integer-clock BG suffix."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import pickle
from random import Random
from time import perf_counter, process_time

from algorithm.arrival_aware_lane_rollout import sample_planning_arrivals
from algorithm import parallel_persistent_current_policy_lane_rollout as original
from algorithm import tick_persistent_current_policy_lane_rollout as optimized
from algorithm.parallel_root_lane_rollout import _semantic_state
from algorithm.experiments import aligned_hardware_exact_reuse_benchmark as anchors
from algorithm.experiments.aligned_hardware_performance_necessity import load_models


ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "md/experiment_artifacts"
STEM = "aligned_hardware_tick_background_rollout_20261002"
PARENT = "aligned_hardware_persistent_current_policy_development_20261002"
SOURCES = ("algorithm/tick_background_rollout.py",
    "algorithm/tick_persistent_current_policy_lane_rollout.py",
    "algorithm/experiments/aligned_hardware_tick_background_benchmark.py")


def artifact(directory, suffix):
    return Path(directory) / (STEM + suffix)


def read(file):
    return json.loads(Path(file).read_text())


def dump(file, value):
    Path(file).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(directory, workers):
    prior = anchors.load_protocol(directory)
    parent = read(Path(directory) / (PARENT + "_protocol.json"))
    for name, literal in parent["execution_sources"].items():
        if (ROOT / name).read_text() != literal:
            raise ValueError("frozen parent source changed: " + name)
    doc = read(Path(directory) / (PARENT + "_shard_08.json"))
    if doc["registered_at_utc"] != parent["registered_at_utc"] or not doc["row"]["rollout_checks_passed"]:
        raise ValueError("completed execution-valid heavy tape required")
    with gzip.open(doc["trajectory_file"], "rt") as source:
        trajectory = json.load(source)["runs"][0]["policies"]["arrival_rollout"]
    tape = parent["tapes"][8]
    model = next(m for m in load_models(parent["models"]) if m.node == tape["node"])
    frames = trajectory["frames"]
    complex_frames = sorted((f["rollout"]["late_current_calls"], i) for i, f in enumerate(frames)
        if f["rollout"]["late_current_calls"] > 0 and not f["rollout"]["planning_overflow"])
    _, wanted = complex_frames[len(complex_frames) // 2]
    seed = f"arrival-plan-v1:{model.node}:{tape['load']}:online"
    samples, cap = parent["planning_samples"], parent["planning_event_cap"]
    rng = Random(seed)
    for frame in frames[:wanted + 1]:
        rates = tuple(v / frame["start"] if frame["start"] else 0.
            for v in frame["observed_arrival_counts"])
        paths = []
        if frame["rollout"]["eligible"]:
            for _ in range(samples):
                paths.append(sample_planning_arrivals(rates, max(o.duration for o in model.options), rng, cap))
                if paths[-1].overflow:
                    break
    if (len(paths) != samples or any(p.overflow for p in paths)
            or sum(len(p.events) for p in paths) != frame["rollout"]["tagged_planning_events"]):
        raise ValueError("heavy frame prefix population differs")
    initial = original.PersistentCurrentPolicyBGIncumbentLaneRollout(model.options, seed, samples, cap)
    payload = {"root": frame["option_indices"][0], "sample_start": 0, "paths": tuple(paths[:2]),
        "queue": tuple(frame["queue"]), "available": tuple(frame["available_before"]),
        "now": frame["start"], "lane": frame["lane_ids"][0],
        "ages": tuple(tuple(g) for g in frame["waiting_ages"]),
        "outer_state": (rng.getstate(), wanted, rates, tuple(frame["observed_arrival_counts"])),
        "input_states": (_semantic_state(initial.inner_planner),
                         _semantic_state(initial.inner_planner.inner_planner))}
    payload_file = artifact(directory, "_heavy_payload.pkl")
    with payload_file.open("wb") as output:
        pickle.dump({"options": model.options, "seed": seed, "samples": samples, "cap": cap,
            "payload": payload}, output)
    protocol = {"registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "planning_samples": samples, "planning_event_cap": cap, "workers": workers,
        "models": prior["models"], "contexts": prior["contexts"],
        "anchor_registration": prior["registered_at_utc"], "parent_registration": parent["registered_at_utc"],
        "heavy": {"payload_file": str(payload_file), "tape_id": tape["tape_id"],
            "frame_index": wanted, "now": frame["start"], "root": payload["root"],
            "queue": frame["queue"], "all_frame_late_calls": frame["rollout"]["late_current_calls"]},
        "execution_sources": {**parent["execution_sources"],
            **{name: (ROOT / name).read_text() for name in SOURCES}},
        "scope": f"Four fixed complete persistent proposals on four real catalogues, paired with the same{workers}-worker setting, plus the previously profiled heavy cold root/two-sample chunk repeated three times in AB/BA/AB order. Keep8/cap128, native choices/exact scores/all logical fees/controller state. No new service trajectory or algorithm-performance claim. Child entry states for the heavy diagnostic use constructors; it is not a replay of the complete factual proposal."}
    dump(artifact(directory, "_protocol.json"), protocol)
    print(json.dumps({"registered_at_utc": protocol["registered_at_utc"],
        "complete_proposals": 4, "heavy_chunk_repeats": 3, "payload_file": str(payload_file)}), flush=True)


def load_protocol(directory):
    protocol = read(artifact(directory, "_protocol.json"))
    for name, literal in protocol["execution_sources"].items():
        if (ROOT / name).read_text() != literal:
            raise ValueError("registered benchmark source changed: " + name)
    return protocol


def run_proposal(index, protocol):
    row = protocol["contexts"][index]
    tape, context = row["tape"], row["context"]
    model = next(m for m in load_models(protocol["models"]) if m.node == tape["node"])
    outputs, timing = {}, {}
    factories = (("original", original.ParallelPersistentCurrentPolicyBGIncumbentLaneRollout),
        ("optimized", optimized.ParallelTickPersistentCurrentPolicyBGIncumbentLaneRollout))
    for label, factory in factories:
        class Recording(factory):
            def score_background_roots(self, *args):
                self.last_scores = super().score_background_roots(*args)
                return self.last_scores

        with Recording(model.options, f"arrival-plan-v1:{model.node}:{tape['load']}:online",
                protocol["planning_samples"], protocol["planning_event_cap"], workers=protocol["workers"]) as planner:
            planner.last_scores = None
            planner.rng.setstate(anchors.rng_tuple(row["state"]["rng_state"]))
            planner.decision_counter = row["state"]["decision_counter"]
            wall, cpu = perf_counter(), process_time()
            decision = planner(model.options, tuple(context["queue"]), tuple(context["available_before"]),
                context["start"], context["lane_ids"][0], tuple(tuple(g) for g in context["waiting_ages"]),
                tuple(context["observed_arrival_counts"]))
            wall, parent_cpu = perf_counter() - wall, process_time() - cpu
            executed = planner.reuse_work
            worker_cpu = executed.get("parallel_root_work", {}).get("worker_cpu_s", 0.)
            timing[label] = {"wall_s": wall, "parent_cpu_s": parent_cpu,
                "worker_cpu_s": worker_cpu, "cpu_s": parent_cpu + worker_cpu,
                "kernel_work": executed["kernel_work"]}
            outputs[label] = (asdict(decision), planner.last_scores, anchors.planner_states(planner))
        print(tape["tape_id"], label, wall, flush=True)
    checks = {"decision_and_logical_work_exact": outputs["original"][0] == outputs["optimized"][0],
        "fraction_scores_exact": outputs["original"][1] == outputs["optimized"][1],
        "controller_states_exact": outputs["original"][2] == outputs["optimized"][2]}
    return {"kind": "complete_proposal", "tape_id": tape["tape_id"], "checks": checks, "timing": timing,
        "wall_speedup": timing["original"]["wall_s"] / timing["optimized"]["wall_s"],
        "cpu_speedup": timing["original"]["cpu_s"] / timing["optimized"]["cpu_s"]}


def run_chunk(protocol):
    with Path(protocol["heavy"]["payload_file"]).open("rb") as source:
        fixture = pickle.load(source)
    repeats = []
    for order in (("original", "optimized"), ("optimized", "original"), ("original", "optimized")):
        results, timing = {}, {}
        for label in order:
            module = original if label == "original" else optimized
            module._initialize_worker(fixture["options"], fixture["seed"], fixture["samples"], fixture["cap"])
            wall, cpu = perf_counter(), process_time()
            result = module._run_root(fixture["payload"])
            timing[label] = {"wall_s": perf_counter() - wall, "cpu_s": process_time() - cpu,
                "kernel_work": result["actual_reuse_work"]["kernel_work"]}
            results[label] = {k: v for k, v in result.items() if k not in (
                "actual_reuse_work", "worker_cpu_s", "worker_wall_s", "constructor_cpu_s", "constructor_wall_s")}
        repeats.append({"order": order, "checks": {"exact_scores_logical_fees_and_states":
            results["original"] == results["optimized"]}, "timing": timing,
            "current_calls": results["original"]["recourse_work"]["recourse_calls"],
            "late_calls": results["original"]["continuation_work"]["late_current_calls"]})
    timing = {label: {key: sum(r["timing"][label][key] for r in repeats) for key in ("wall_s", "cpu_s")}
        for label in ("original", "optimized")}
    return {"kind": "heavy_cold_chunk", **protocol["heavy"], "repeats": repeats,
        "checks": {"all_repeats_exact": all(all(r["checks"].values()) for r in repeats)}, "timing": timing,
        "wall_speedup": timing["original"]["wall_s"] / timing["optimized"]["wall_s"],
        "cpu_speedup": timing["original"]["cpu_s"] / timing["optimized"]["cpu_s"]}


def run(index, directory):
    protocol = load_protocol(directory)
    record = run_chunk(protocol) if index == 4 else run_proposal(index, protocol)
    record.update(index=index, registered_at_utc=protocol["registered_at_utc"])
    dump(artifact(directory, f"_{index:02d}.json"), record)
    if not all(record["checks"].values()):
        raise ValueError("tick kernel changes native scores/work/state: " + str(index))
    print(json.dumps({"index": index, "checks_passed": True, "wall_speedup": record["wall_speedup"]}), flush=True)


def merge(directory):
    protocol = load_protocol(directory)
    records = [read(artifact(directory, f"_{i:02d}.json")) for i in range(5)]
    if any(r["index"] != i or r["registered_at_utc"] != protocol["registered_at_utc"] for i, r in enumerate(records)):
        raise ValueError("five matching benchmark records required")
    totals = {label: {key: sum(r["timing"][label][key] for r in records[:4]) for key in ("wall_s", "cpu_s")}
        for label in ("original", "optimized")}
    passed = all(all(r["checks"].values()) for r in records)
    result = {"decision": "TICK_BACKGROUND_BENCHMARK_" + ("PASS" if passed else "FAIL"),
        "registered_at_utc": protocol["registered_at_utc"], "exact_match_passed": passed,
        "records": records, "complete_proposal_totals": totals,
        "pooled_proposal_wall_speedup": totals["original"]["wall_s"] / totals["optimized"]["wall_s"],
        "pooled_proposal_cpu_speedup": totals["original"]["cpu_s"] / totals["optimized"]["cpu_s"],
        "new_full_trajectories": 0, "scope": protocol["scope"]}
    dump(artifact(directory, "_summary.json"), result)
    print(json.dumps({k: result[k] for k in ("decision", "pooled_proposal_cpu_speedup", "pooled_proposal_wall_speedup")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", type=int, choices=range(5))
    mode.add_argument("--merge", action="store_true")
    parser.add_argument("--artifact-dir", type=Path, default=DIRECTORY)
    parser.add_argument("--workers", type=int, default=13)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.artifact_dir, args.workers)
    elif args.run is not None:
        run(args.run, args.artifact_dir)
    else:
        merge(args.artifact_dir)


if __name__ == "__main__":
    main()
