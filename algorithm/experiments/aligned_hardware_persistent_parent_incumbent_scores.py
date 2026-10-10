"""Score one offline parent-incumbent rule at all24 first divergences."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import pickle
import socket
from time import perf_counter

from algorithm.arrival_aware_lane_rollout import sample_planning_arrivals
from algorithm.parallel_root_lane_rollout import _restore_semantic, _semantic_state
from algorithm.experiments import aligned_hardware_tick_background_benchmark as tick
from algorithm.experiments.aligned_hardware_performance_necessity import load_models
from algorithm.experiments.aligned_hardware_two_step_current_policy_root_ranking import same
from algorithm.experiments.persistent_continuation_cross_selector import capture_planner_tree
from algorithm.experiments.persistent_current_policy_checks import check_rollout
from algorithm.experiments.persistent_parent_incumbent_scores import (
    RecordingParallelPersistentCurrentPolicyScores, choose_parent_incumbent,
    last_touched_frames, recover_frame_child_states,
)
from algorithm.experiments.two_step_bg_incumbent_continuation_cross_replay import (
    reconstruct_prefix_states, restore_planner_state,
)

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
PARENT = "aligned_hardware_persistent_current_policy_development_20261002"
DIVERGENCE = "aligned_hardware_persistent_first_divergence_20261002"
STEM = "aligned_hardware_persistent_parent_incumbent_scores_20261002"
SOURCES = ("algorithm/experiments/aligned_hardware_persistent_parent_incumbent_scores.py",
    "algorithm/experiments/persistent_parent_incumbent_scores.py")


def path(directory, suffix):
    return Path(directory) / (STEM + suffix)


def read(file):
    return json.loads(Path(file).read_text())


def dump(file, data):
    with Path(file).open("x") as output:
        json.dump(data, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def register(directory, workers):
    parent = read(Path(directory) / (PARENT + "_protocol.json"))
    completed = read(Path(directory) / (PARENT + "_summary.json"))
    divergence = read(Path(directory) / (DIVERGENCE + "_summary.json"))
    optimized = tick.load_protocol(directory)
    benchmark = read(tick.artifact(directory, "_summary.json"))
    require(completed["execution_checks_passed"] and completed["completed_shards"] == 24
        and completed["registered_at_utc"] == parent["registered_at_utc"]
        and completed["decision"] == "PERSISTENT_CURRENT_POLICY_DEVELOPMENT_HOLD"
        and divergence["source_registered_at_utc"] == parent["registered_at_utc"]
        and divergence["finite_scientific_registered_at_utc"] == parent["finite_scientific_registered_at_utc"]
        and divergence["pairs"] == divergence["divergent_pairs"] == 24
        and benchmark["exact_match_passed"]
        and benchmark["registered_at_utc"] == optimized["registered_at_utc"]
        and optimized["parent_registration"] == parent["registered_at_utc"]
        and parent["planning_samples"] == 8 and parent["planning_event_cap"] == 128,
        "completed frozen24 HOLD, first divergences and exact Tick execution required")
    assignments = []
    for i, (tape, record) in enumerate(zip(parent["tapes"], divergence["records"], strict=True)):
        doc = read(Path(directory) / (PARENT + f"_shard_{i:02d}.json"))
        first = record["first_divergence"]
        require(doc["index"] == i and doc["registered_at_utc"] == parent["registered_at_utc"]
            and doc["row"]["rollout_checks_passed"]
            and doc["row"]["tape_id"] == record["tape_id"] == tape["tape_id"]
            and len(first["finite"]["option_indices"]) == len(first["persistent"]["option_indices"]) == 1,
            "all24 literal first-divergence identities required")
        assignments.append(dict(index=i, **{k: tape[k] for k in ("tape_id", "node", "lanes", "load", "seed")},
            frame_index=first["frame_index"], first_divergence=first, trajectory_file=doc["trajectory_file"]))
    sources = {**parent["execution_sources"], **optimized["execution_sources"],
        **{name: (ROOT / name).read_text() for name in SOURCES}}
    for name, literal in sources.items():
        require((ROOT / name).read_text() == literal, "frozen source differs: " + name)
    protocol = {k: parent[k] for k in ("models", "planning_samples", "planning_event_cap")}
    protocol.update(registered_at_utc=datetime.now(timezone.utc).isoformat(), workers=workers,
        full24_registered_at_utc=parent["registered_at_utc"], full24_decision=completed["decision"],
        finite_scientific_registered_at_utc=parent["finite_scientific_registered_at_utc"],
        tick_registered_at_utc=optimized["registered_at_utc"], assignments=assignments,
        rule="Keep original BG S2<=MW risk set. Let p be the cached finite-parent action at the same first-divergence state. Replace p only for strictly lower persistent S1 and persistent S2<=S2(p). Parent wins S1 ties; strict candidates use duration/index ties. Exact Fraction comparisons,8 samples/cap128.",
        state_initialization="Reconstruct causal outer prefix sampling and counter from canonical births/cached FIFO frames. Recover each child's last included touched state from native root/two-sample chunks at its last-touched frame. First virtual calls reset RNG/counter/rates; untouched outputs are never restored. Normalize outer rates/counts to anchor context before its public call. No constructor substitution for previously touched children.",
        execution="Register metadata only. Each of24 shards restores the true semantic prefix tree, records one complete native target proposal without rescoring either BG/main stage, and checks every cached native decision/fee field. Prefix recovery physical work is recorded separately. No service suffix or fresh full24 rollout.",
        interpretation="Offline development action diagnostic. Cached finite-parent action is a reference, not a deployable oracle: a production chooser would compute it live. Full24 HOLD and sealed confirmation unchanged; action changes establish no service-performance gain.",
        execution_sources=sources)
    dump(path(directory, "_protocol.json"), protocol)
    print(json.dumps(dict(registered_at_utc=protocol["registered_at_utc"], states=24, workers=workers)), flush=True)


def load_protocol(directory):
    protocol = read(path(directory, "_protocol.json"))
    for name, literal in protocol["execution_sources"].items():
        require((ROOT / name).read_text() == literal, "registered source changed: " + name)
    return protocol


def prepare_prefix(planner, factual, anchor, work):
    frames = factual["frames"]
    last = last_touched_frames(frames, anchor)
    wanted = {anchor, *(i for i in last if i is not None)}
    states = reconstruct_prefix_states(planner, {**factual, "frames": frames[:anchor + 1]}, wanted, work)
    for index in sorted(wanted - {anchor}):
        frame = frames[index]
        work.update(restore_planner_state(planner, states[index]))
        planner.arrival_rates = tuple(frame["rollout"]["estimated_arrival_rates"])
        planner._outer_observed_counts = tuple(frame["observed_arrival_counts"])
        paths = tuple(sample_planning_arrivals(planner.arrival_rates, planner.maximum_duration,
            planner.rng, planner.event_cap) for _ in range(planner.sample_count))
        work["recovery_prefix_sample_calls"] += len(paths)
        work["recovery_prefix_random_draws"] += sum(p.random_draws for p in paths)
        work["recovery_prefix_rate_reads"] += sum(p.rate_reads for p in paths)
        work["recovery_prefix_events"] += sum(len(p.events) for p in paths)
        require(not any(p.overflow for p in paths), "last-touched prefix unexpectedly overflowed")
        needed = tuple(level for level, touched in enumerate(last) if touched == index)
        outer_state = _semantic_state(planner)
        work["recovery_outer_state_reads"] += 1
        work["recovery_outer_rng_words_copied"] += len(outer_state[0][1])
        work["recovery_outer_rate_reads"] += len(outer_state[2])
        work["recovery_outer_count_reads"] += len(outer_state[3])
        recovered, fees = recover_frame_child_states(planner, frame, paths, outer_state, needed)
        work.update(fees)
        for level, state in recovered.items():
            child = planner.inner_planner if level == 0 else planner.inner_planner.inner_planner
            _restore_semantic(child, state)
            work["recovered_child_state_writes"] += 1
            work["recovered_child_rng_words_restored"] += len(state[0][1])
            work["recovered_child_rate_writes"] += len(state[2])
            work["recovered_child_count_writes"] += len(state[3]) if state[3] is not None else 0
        print(json.dumps(dict(phase="prefix_child_restored", frame_index=index, levels=needed)), flush=True)
    work.update(restore_planner_state(planner, states[anchor]))
    planner.arrival_rates = tuple(frames[anchor]["rollout"]["estimated_arrival_rates"])
    planner._outer_observed_counts = tuple(frames[anchor]["observed_arrival_counts"])
    return last, capture_planner_tree(planner, work)


def run_shard(index, directory):
    protocol = load_protocol(directory)
    target = protocol["assignments"][index]
    started = perf_counter()
    print(json.dumps(dict(phase="read_factual_prefix", index=index, tape_id=target["tape_id"])), flush=True)
    with gzip.open(target["trajectory_file"], "rt") as source:
        saved = json.load(source)
    require(saved["registered_at_utc"] == protocol["full24_registered_at_utc"], "factual epoch differs")
    factual = next(r["policies"]["arrival_rollout"] for r in saved["runs"] if r["tape_id"] == target["tape_id"])
    frame = factual["frames"][target["frame_index"]]
    first = target["first_divergence"]
    require(frame["option_indices"] == first["persistent"]["option_indices"]
        and frame["start"] == first["start"] and frame["queue"] == first["queue"], "cached anchor differs")
    model = next(m for m in load_models(protocol["models"]) if m.node == target["node"])
    seed = f"arrival-plan-v1:{model.node}:{target['load']}:online"
    work = Counter()
    with RecordingParallelPersistentCurrentPolicyScores(model.options, seed, protocol["planning_samples"],
            protocol["planning_event_cap"], workers=protocol["workers"]) as planner:
        last, before = prepare_prefix(planner, factual, target["frame_index"], work)
        print(json.dumps(dict(phase="native_target_proposal", index=index, current=0, total=1)), flush=True)
        decision = planner(model.options, tuple(frame["queue"]), tuple(frame["available_before"]),
            frame["start"], frame["lane_ids"][0], tuple(map(tuple, frame["waiting_ages"])),
            tuple(frame["observed_arrival_counts"]))
        info = asdict(decision)
        selected = info.pop("selection")
        require(same(info, frame["rollout"]) and all(same(selected[k], frame[k]) for k in
            ("option_indices", "service", "duration", "score", "upper_bound", "gap", "score_exact", "canonical_exact"))
            and planner.decision_counter == target["frame_index"] + 1
            and check_rollout({"frames": [frame]}, model, protocol, planner), "native decision/fees did not reproduce")
        original, scored = planner.recorded_original_scores, planner.recorded_scores
        require(original is not None and scored is not None
            and not original.background_rollout_overflow and not scored.background_rollout_overflow,
            "complete native original/main score objects required")
        p, = first["finite"]["option_indices"]
        native, = frame["option_indices"]
        mw, bg = info["incumbent_maxweight_root_index"], info["incumbent_root_index"]
        chosen = choose_parent_incumbent(scored, original, p, mw, planner.durations)
        risk = [root for root in scored.costs if original.squared_costs[root] <= original.squared_costs[mw]]
        allowed = [root for root in risk if scored.squared_costs[root] <= scored.squared_costs[p]]
        require(chosen in allowed and (chosen == p or scored.costs[chosen] < scored.costs[p]), "fixed exact rule failed")
        scores = [dict(root=root, original_risk_allowed=root in risk, parent_s2_allowed=root in allowed,
            original_S1_mean=float(original.costs[root] / planner.sample_count),
            original_S2_mean=float(original.squared_costs[root] / planner.sample_count),
            persistent_S1_mean=float(scored.costs[root] / planner.sample_count),
            persistent_S2_mean=float(scored.squared_costs[root] / planner.sample_count)) for root in scored.costs]
        evidence = path(directory, f"_shard_{index:02d}_exact.pkl")
        with evidence.open("xb") as output:
            pickle.dump(dict(registered_at_utc=protocol["registered_at_utc"], target=target,
                before_tree=before, after_tree=capture_planner_tree(planner, work),
                original_scores=original, persistent_scores=scored), output)
        row = dict(index=index, registered_at_utc=protocol["registered_at_utc"],
            **{k: target[k] for k in ("tape_id", "node", "lanes", "load", "seed", "frame_index")},
            execution_checks_passed=True, original_risk_roots=risk, parent_s2_roots=allowed,
            finite_parent_root=p, native_persistent_root=native, original_BG_root=bg, MW_root=mw,
            offline_root=chosen, changed_from_native=chosen != native, retained_parent=chosen == p,
            returned_BG=chosen == bg, last_touched_frames=last, scores=scores,
            native_logical_work=info, prefix_preparation_work=dict(work),
            native_physical_work=planner.reuse_work, recording_work=dict(planner.recording_work),
            parent_process_planner_constructors=3, parent_process_preprocessing_work=dict(planner.preprocessing_work),
            exact_evidence_file=str(evidence), execution_host=socket.gethostname(), wall_s=perf_counter() - started)
    dump(path(directory, f"_shard_{index:02d}.json"), row)
    print(json.dumps(dict(phase="complete", index=index, current=1, total=1,
        finite_parent_root=p, native_root=native, offline_root=chosen, wall_s=row["wall_s"])), flush=True)


def merge(directory):
    protocol = load_protocol(directory)
    rows = [read(path(directory, f"_shard_{i:02d}.json")) for i in range(24)]
    for i, row in enumerate(rows):
        require(row["index"] == i and row["registered_at_utc"] == protocol["registered_at_utc"]
            and row["execution_checks_passed"] and row["tape_id"] == protocol["assignments"][i]["tape_id"],
            "all24 registered native scoring shards required")
    fields = ("changed_from_native", "retained_parent", "returned_BG")
    compact_keys = ("index", "tape_id", "node", "lanes", "load", "seed", "frame_index", "finite_parent_root",
        "native_persistent_root", "original_BG_root", "MW_root", "offline_root", *fields,
        "original_risk_roots", "parent_s2_roots", "last_touched_frames", "scores", "wall_s")
    summary = dict(registered_at_utc=protocol["registered_at_utc"], execution_checks_passed=True,
        pairs=24, completed_shards=24, native_target_proposals=24, service_suffix_executions=0,
        decision="OFFLINE_PARENT_INCUMBENT_SCORES_COMPLETE", full24_decision=protocol["full24_decision"],
        full24_registered_at_utc=protocol["full24_registered_at_utc"], performance_gain_established=False,
        counts={key: sum(r[key] for r in rows) for key in fields},
        high_gpu2_counts={key: sum(r[key] for r in rows if r["node"] == "gpu_b" and r["load"] == .95) for key in fields},
        records=[{key: row[key] for key in compact_keys} for row in rows], interpretation=protocol["interpretation"])
    dump(path(directory, "_summary.json"), summary)
    print(json.dumps({key: summary[key] for key in ("decision", "pairs", "counts", "high_gpu2_counts")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--shard-index", type=int, choices=range(24))
    mode.add_argument("--merge", action="store_true")
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACTS)
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()
    if args.register:
        register(args.artifact_dir, args.workers)
    elif args.shard_index is not None:
        run_shard(args.shard_index, args.artifact_dir)
    else:
        merge(args.artifact_dir)


if __name__ == "__main__":
    main()
