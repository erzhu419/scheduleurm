"""Exact finite-parent S1 ties/headroom at every FIFO mean-guard divergence."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import json
from pathlib import Path
import pickle
import socket
from time import perf_counter, process_time

from algorithm.experiments.aligned_hardware_finite_policy_tagged_ranking import RecordingFiniteScores
from algorithm.experiments.aligned_hardware_persistent_action_controller_cross import prepare_finite_prefix, same
from algorithm.experiments.aligned_hardware_performance_necessity import load_models
from algorithm.experiments.aligned_hardware_persistent_continuation_cross import native_first_frame
from algorithm.experiments.aligned_hardware_two_step_current_policy_bg_incumbent_development import check_rollout
from algorithm.experiments.persistent_continuation_cross_selector import capture_planner_tree

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
STEM = "aligned_hardware_fifo_s1_headroom_20261003"
PREVIOUS = "aligned_hardware_fifo_mean_guard_development_20261003"
SOURCE = str(Path(__file__).resolve().relative_to(ROOT))
SOURCES = (SOURCE, "algorithm/experiments/aligned_hardware_finite_policy_tagged_ranking.py",
    "algorithm/experiments/aligned_hardware_persistent_action_controller_cross.py",
    "algorithm/experiments/aligned_hardware_persistent_continuation_cross.py",
    "algorithm/experiments/two_step_bg_incumbent_continuation_cross_replay.py",
    "algorithm/experiments/persistent_parent_incumbent_scores.py",
    "algorithm/experiments/persistent_continuation_cross_selector.py",
    "skill/tests/test_fifo_s1_headroom.py")


def path(directory, suffix):
    return Path(directory) / (STEM + suffix)


def read(file):
    return json.loads(Path(file).read_text())


def dump(file, data):
    with Path(file).open("x") as out:
        json.dump(data, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write("\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact(value):
    return [value.numerator, value.denominator] if value is not None else None


def root_screen(scored, original, parent_root, guard_root, bg_root, mw_root, sample_count=8):
    """Compare exact S1 within the original BG S2<=MW set, without new scoring."""
    require(not scored.background_rollout_overflow and not original.background_rollout_overflow
        and set(scored.costs) == set(original.costs) and sample_count == 8,
        "complete identical native root sets and eight paths required")
    values = {root: (scored.costs[root], scored.squared_costs[root], original.squared_costs[root])
              for root in scored.costs}
    require(all(isinstance(value, Fraction) for row in values.values() for value in row),
        "native exact Fraction scores required")
    limit = values[mw_root][2]
    risk = [root for root, row in values.items() if row[2] <= limit]
    require(parent_root in risk and guard_root in risk and bg_root in risk, "registered roots outside original risk set")
    minimum = min(values[root][0] for root in risk)
    argmin = [root for root in risk if values[root][0] == minimum]
    require(parent_root in argmin, "cached finite-parent root is not a genuine S1 minimizer")
    runner_up = min((values[root][0] for root in risk if values[root][0] > minimum), default=None)
    measures = {"min_S1": minimum, "BG_headroom": values[bg_root][0] - minimum,
        "guard_gap": values[guard_root][0] - minimum,
        "runner_up_gap": None if runner_up is None else runner_up - minimum}
    return {"original_risk_roots": risk, "argmin_roots": argmin, "exact_tie_count": len(argmin),
        **{name + "_exact": exact(value) for name, value in measures.items()},
        **{name + "_per_path": float(value / sample_count) if value is not None else None
           for name, value in measures.items()},
        "roots": [{"root": root, "risk_allowed": root in risk,
            "S1_exact": exact(row[0]), "S2_exact": exact(row[1]), "original_S2_exact": exact(row[2]),
            "S1_per_path": float(row[0] / sample_count), "S2_per_path": float(row[1] / sample_count)}
            for root, row in values.items()],
        "diagnostic_work": {"native_root_score_reads": 3 * len(values), "recorded_root_rows": len(values)}}


def register(directory, workers=13):
    previous = read(Path(directory) / (PREVIOUS + "_protocol.json"))
    summary = read(Path(directory) / (PREVIOUS + "_summary.json"))
    require(summary["execution_checks_passed"] and summary["completed_shards"] == 24
        and summary["registered_at_utc"] == previous["registered_at_utc"]
        and previous["planning_samples"] == 8 and previous["planning_event_cap"] == 128
        and workers == 13 and len(previous["tapes"]) == 24, "completed frozen24 mean-guard source required")
    inventory = []
    for i, (tape, reference) in enumerate(zip(previous["tapes"], previous["references"], strict=True)):
        doc = read(Path(directory) / (PREVIOUS + f"_shard_{i:02d}.json"))
        require(doc["index"] == i and doc["registered_at_utc"] == previous["registered_at_utc"]
            and doc["row"]["tape_id"] == reference["tape_id"] == tape["tape_id"]
            and doc["row"]["rollout_checks_passed"], "cached mean-guard shard identity differs")
        inventory.append({"source_index": i, **{k: tape[k] for k in ("tape_id", "node", "load", "seed")},
            "first_divergence": doc["row"]["action_comparison"]["first_divergence"], "reference": reference})
    assignments = [dict(index=i, **row) for i, row in enumerate(r for r in inventory if r["first_divergence"] is not None)]
    require(len(assignments) == summary["divergent_tapes"], "retain every divergence without outcome filtering")
    sources = {**previous["execution_sources"], **{name: (ROOT / name).read_text() for name in SOURCES}}
    for name, literal in sources.items():
        require((ROOT / name).read_text() == literal, "frozen source changed: " + name)
    protocol = {k: previous[k] for k in ("models", "planning_samples", "planning_event_cap", "finite_scientific_registered_at_utc")}
    protocol.update(registered_at_utc=datetime.now(timezone.utc).isoformat(), workers=workers,
        source_registered_at_utc=previous["registered_at_utc"], inventory=inventory, assignments=assignments,
        execution_sources=sources, expected_native_proposals_upper_bound=3 * len(assignments),
        scope="All mean-guard first divergences on actual finite-parent states; retain all24 inventory, including no-divergence tapes. Recover complete finite own prefix and consume one native anchor proposal. Exact S1 ties/headroom only; no waiting/J rescoring, production rule, suffix trajectories or policy-gain claim.")
    dump(path(directory, "_protocol.json"), protocol)
    print(json.dumps({"registered_at_utc": protocol["registered_at_utc"], "inventory": len(inventory),
        "shards": len(assignments), "workers": workers}), flush=True)


def load_protocol(directory):
    protocol = read(path(directory, "_protocol.json"))
    for name, literal in protocol["execution_sources"].items():
        require((ROOT / name).read_text() == literal, "registered source changed: " + name)
    return protocol


def run_shard(index, directory):
    protocol = load_protocol(directory)
    target = protocol["assignments"][index]
    reference, first = target["reference"], target["first_divergence"]
    source = read(Path(directory) / (PREVIOUS + f"_shard_{target['source_index']:02d}.json"))
    require(source["registered_at_utc"] == protocol["source_registered_at_utc"]
        and source["row"]["action_comparison"]["first_divergence"] == first, "frozen divergence changed")
    ref_doc = read(reference["row_path"])
    require(ref_doc["tape_id"] == target["tape_id"] and ref_doc["trajectory_file"] == reference["trajectory_file"]
        and ref_doc["scientific_registered_at_utc"] == protocol["finite_scientific_registered_at_utc"]
        and ref_doc["distribution_registered_at_utc"] == reference["distribution_registered_at_utc"], "finite reference identity differs")
    with gzip.open(reference["trajectory_file"], "rt") as source:
        saved = json.load(source)
    require(saved["registered_at_utc"] == protocol["finite_scientific_registered_at_utc"], "finite trajectory epoch differs")
    factual = next(row["policies"]["arrival_rollout"] for row in saved["runs"] if row["tape_id"] == target["tape_id"])
    frame = factual["frames"][first["frame_index"]]
    require(frame["start"] == first["start"] and frame["queue"] == first["queue"]
        and frame["option_indices"] == [first["parent_root"]], "cached finite anchor differs")
    model = next(m for m in load_models(protocol["models"]) if m.node == target["node"])
    seed = f"arrival-plan-v1:{model.node}:{target['load']}:online"
    work, cpu, wall = Counter(), process_time(), perf_counter()
    with RecordingFiniteScores(model.options, seed, 8, 128, workers=protocol["workers"]) as planner:
        last, before = prepare_finite_prefix(planner, factual, first["frame_index"], work)
        decision = planner(model.options, tuple(frame["queue"]), tuple(frame["available_before"]),
            frame["start"], frame["lane_ids"][0], tuple(map(tuple, frame["waiting_ages"])),
            tuple(frame["observed_arrival_counts"]))
        require(same(native_first_frame(frame, decision), frame)
            and check_rollout({"frames": [frame]}, model, protocol, planner)
            and decision.selection.option_indices == (first["parent_root"],), "complete native finite anchor did not reproduce")
        scored, original = planner.recorded_scores, planner.recorded_original_scores
        require(scored is not None and original is not None and len(planner.recorded_paths) == 8,
            "complete native score recording required")
        screen = root_screen(scored, original, first["parent_root"], first["candidate_root"],
            decision.incumbent_root_index, decision.incumbent_maxweight_root_index)
        after = capture_planner_tree(planner, work)
        require(len(before["planners"]) == len(after["planners"]) == 3
            and after["planners"][0]["decision_counter"] == before["planners"][0]["decision_counter"] + 1,
            "full finite semantic tree counters differ")
        require(work["finite_prefix_native_proposals"] <= 2, "prefix recovery exceeded two native proposals")
        with path(directory, f"_prepared_{index:02d}.pkl").open("xb") as output:
            pickle.dump(dict(registered_at_utc=protocol["registered_at_utc"], target=target,
                options=model.options, seed=seed, frame=frame, paths=planner.recorded_paths,
                before_tree=before, after_tree=after, native_scores=scored, original_scores=original), output)
        physical = planner.reuse_work
        row = dict(index=index, registered_at_utc=protocol["registered_at_utc"],
            **{k: target[k] for k in ("tape_id", "node", "load", "seed")}, frame_index=first["frame_index"],
            execution_checks_passed=True, parent_root=first["parent_root"], guard_root=first["candidate_root"],
            BG_root=decision.incumbent_root_index, MW_root=decision.incumbent_maxweight_root_index,
            prefix_native_proposals=work["finite_prefix_native_proposals"], anchor_native_proposals=1,
            before_counters=[p["decision_counter"] for p in before["planners"]],
            after_counters=[p["decision_counter"] for p in after["planners"]], last_touched_frames=last,
            screen=screen, prefix_and_state_work=dict(work), anchor_native_work=asdict(decision),
            recording_work=dict(planner.recording_work), physical_work=physical,
            controller_cpu_s=process_time() - cpu, worker_cpu_s=physical["parallel_root_work"].get("worker_cpu_s", 0.),
            wall_s=perf_counter() - wall, execution_host=socket.gethostname())
        row["charged_cpu_s"] = row["controller_cpu_s"] + row["worker_cpu_s"]
    dump(path(directory, f"_shard_{index:02d}.json"), row)
    print(json.dumps({"index": index, "complete": True, "ties": screen["exact_tie_count"],
        "guard_gap": screen["guard_gap_per_path"], "native_proposals": row["prefix_native_proposals"] + 1}), flush=True)


def merge(directory):
    protocol = load_protocol(directory)
    rows = []
    for target in protocol["assignments"]:
        row = read(path(directory, f"_shard_{target['index']:02d}.json"))
        require(row["index"] == target["index"] and row["tape_id"] == target["tape_id"]
            and row["registered_at_utc"] == protocol["registered_at_utc"] and row["execution_checks_passed"],
            "all registered exact screens required")
        rows.append(row)
    totals = {key: sum(row[key] for row in rows) for key in
        ("prefix_native_proposals", "anchor_native_proposals", "charged_cpu_s", "controller_cpu_s", "worker_cpu_s")}
    summary = dict(decision="FIFO_S1_HEADROOM_SCREEN_COMPLETE", execution_checks_passed=True,
        performance_gain_established=False, registered_at_utc=protocol["registered_at_utc"],
        inventory_tapes=len(protocol["inventory"]), screened_states=len(rows),
        nondivergent_tapes=[r["tape_id"] for r in protocol["inventory"] if r["first_divergence"] is None],
        states_with_multiple_minima=sum(row["screen"]["exact_tie_count"] > 1 for row in rows),
        guard_above_minimum_states=sum(row["screen"]["guard_gap_exact"][0] > 0 for row in rows),
        totals=totals, diagnostic_work={key: sum(row["screen"]["diagnostic_work"][key] for row in rows)
            for key in ("native_root_score_reads", "recorded_root_rows")},
        recording_work={key: sum(row["recording_work"].get(key, 0) for row in rows)
            for key in sorted({key for row in rows for key in row["recording_work"]})},
        records=[{key: row[key] for key in ("tape_id", "node", "load", "seed", "frame_index", "parent_root",
            "guard_root", "BG_root", "MW_root", "before_counters", "after_counters", "screen")} for row in rows],
        interpretation="Exact native predicted-S1 headroom on all frozen first-divergence states; this is neither a representative whole-state survey nor a policy performance gain. No production min-guard or new service trajectories were executed.")
    dump(path(directory, "_summary.json"), summary)
    print(json.dumps({key: summary[key] for key in ("decision", "screened_states", "states_with_multiple_minima",
        "guard_above_minimum_states", "totals")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--register", action="store_true")
    modes.add_argument("--run-shard", action="store_true")
    modes.add_argument("--merge", action="store_true")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--workers", type=int, default=13)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    if args.register:
        register(args.artifact_dir, args.workers)
    elif args.run_shard:
        require(args.shard_index is not None, "--run-shard requires --shard-index")
        run_shard(args.shard_index, args.artifact_dir)
    else:
        merge(args.artifact_dir)


if __name__ == "__main__":
    main()
