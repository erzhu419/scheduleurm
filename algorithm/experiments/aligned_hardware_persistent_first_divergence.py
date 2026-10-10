"""Read cached full24 trajectories; locate the first physical action divergence."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PARENT = "aligned_hardware_persistent_current_policy_development_20261002"
STEM = "aligned_hardware_persistent_first_divergence_20261002"
CONTEXT = ("start", "queue", "waiting_ages", "available_before", "observed_arrival_counts", "lane_ids")
PREFIX = ("eligible", "queue_limit", "window_s", "sample_count", "configured_sample_count",
    "tagged_planning_events", "estimated_arrival_rates", "rate_exposure_s")
ORIGINAL = ("incumbent_maxweight_root_index", "incumbent_root_index", "incumbent_changed",
    "incumbent_candidate_count", "incumbent_risk_feasible_candidates", "incumbent_risk_filtered_candidates",
    "incumbent_risk_comparisons", "incumbent_mw_predicted_completion_s", "incumbent_predicted_completion_s",
    "incumbent_mw_predicted_squared_flow_s2", "incumbent_predicted_squared_flow_s2",
    "incumbent_background_rollout_overflow")
FORECAST = ("baseline_predicted_completion_s", "chosen_predicted_completion_s",
    "baseline_predicted_squared_flow_s2", "chosen_predicted_squared_flow_s2",
    "incumbent_actual_predicted_squared_flow_s2", "planning_overflow", "background_rollout_overflow")


def read(file):
    return json.loads(Path(file).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def available(info):
    return (info["eligible"] and not info["planning_overflow"]
        and info["chosen_predicted_squared_flow_s2"] is not None
        and info["baseline_predicted_squared_flow_s2"] is not None)


def exceeds(info):
    return info["chosen_predicted_squared_flow_s2"] > info["baseline_predicted_squared_flow_s2"]


def summarize(directory, output):
    directory = Path(directory)
    protocol = read(directory / (PARENT + "_protocol.json"))
    parent = read(directory / (PARENT + "_summary.json"))
    epoch = protocol["registered_at_utc"]
    require(parent["execution_checks_passed"] and parent["completed_shards"] == 24
        and parent["registered_at_utc"] == epoch
        and parent["decision"] == "PERSISTENT_CURRENT_POLICY_DEVELOPMENT_HOLD", "completed full24 HOLD required")
    require(len(protocol["tapes"]) == len(protocol["references"]) == 24, "all24 literal pairs required")
    documents = [read(directory / (PARENT + f"_shard_{i:02d}.json")) for i in range(24)]
    uses = Counter(str(Path(p).resolve()) for d, r in zip(documents, protocol["references"])
        for p in (d["trajectory_file"], r["trajectory_file"]))
    cache = {}

    def frames(file, registration, tape_id):
        key = str(Path(file).resolve())
        if key in cache:
            saved_epoch, runs = cache[key]
        else:
            with gzip.open(key, "rt") as source:
                saved = json.load(source)
            saved_epoch = saved["registered_at_utc"]
            runs = {r["tape_id"]: r["policies"]["arrival_rollout"]["frames"] for r in saved["runs"]}
            if uses[key] > 1:
                cache[key] = saved_epoch, runs
        require(saved_epoch == registration, "trajectory registration differs: " + key)
        return runs[tape_id]

    records = []
    for i, (tape, ref, doc) in enumerate(zip(protocol["tapes"], protocol["references"], documents)):
        tape_id = tape["tape_id"]
        require(doc["index"] == i and doc["registered_at_utc"] == epoch
            and doc["row"]["tape_id"] == ref["tape_id"] == tape_id
            and doc["row"]["rollout_checks_passed"] and ref["cached_checks_passed"], "shard identity/checks differ")
        old_doc = read(ref["row_path"])
        require(old_doc["tape_id"] == tape_id and old_doc["trajectory_file"] == ref["trajectory_file"]
            and old_doc["scientific_registered_at_utc"] == protocol["finite_scientific_registered_at_utc"]
            and old_doc["distribution_registered_at_utc"] == ref["distribution_registered_at_utc"], "finite reference differs")
        old = frames(ref["trajectory_file"], protocol["finite_scientific_registered_at_utc"], tape_id)
        new = frames(doc["trajectory_file"], epoch, tape_id)
        first, stop = None, len(new)
        for j, (a, b) in enumerate(zip(old, new)):
            require(all(a[k] == b[k] for k in CONTEXT), f"context differs before action divergence: {tape_id}/{j}")
            ra, rb = a["rollout"], b["rollout"]
            require(all(ra[k] == rb[k] for k in PREFIX + ORIGINAL), f"prefix/original BG differs: {tape_id}/{j}")
            require(all(ra[k] - ra["background_" + k] == rb[k] - rb["background_" + k]
                for k in ("planning_random_draws", "planning_rate_reads")), f"outer prefix fees differ: {tape_id}/{j}")
            if a["option_indices"] != b["option_indices"]:
                stop = j
                first = dict(frame_index=j, start=b["start"], queue=b["queue"],
                    original_BG_MW={k: rb[k] for k in ORIGINAL},
                    chosen_S2_exceeds_baseline_S2=exceeds(rb) if available(rb) else None,
                    finite={**{k: a[k] for k in ("option_indices", "service", "duration")}, **{k: ra[k] for k in FORECAST}},
                    persistent={**{k: b[k] for k in ("option_indices", "service", "duration")}, **{k: rb[k] for k in FORECAST}})
                break
            require(a["service"] == b["service"] and a["duration"] == b["duration"], f"same action has different service: {tape_id}/{j}")
        if first is None:
            require(len(old) == len(new), "same-action trajectories have unequal lengths: " + tape_id)
        scored = [f["rollout"] for f in new if available(f["rollout"])]
        prior = [f["rollout"] for f in new[:stop] if available(f["rollout"])]
        records.append({**{k: tape[k] for k in ("tape_id", "node", "lanes", "load", "seed")},
            "finite_distribution_registered_at_utc": ref["distribution_registered_at_utc"],
            "finite_frames": len(old), "persistent_frames": len(new), "first_divergence": first,
            "candidate_available_frames": len(scored), "candidate_S2_exceeds_baseline_frames": sum(map(exceeds, scored)),
            "strict_prior_available_frames": len(prior), "strict_prior_S2_exceeds_baseline_frames": sum(map(exceeds, prior))})
    result = dict(source_registered_at_utc=epoch, finite_scientific_registered_at_utc=protocol["finite_scientific_registered_at_utc"],
        source_decision=parent["decision"], source_execution_checks_passed=True, pairs=24,
        divergent_pairs=sum(r["first_divergence"] is not None for r in records),
        first_divergence_S2_exceeds_baseline_pairs=sum(r["first_divergence"] is not None
            and r["first_divergence"]["chosen_S2_exceeds_baseline_S2"] is True for r in records),
        totals={k: sum(r[k] for r in records) for k in ("candidate_available_frames", "candidate_S2_exceeds_baseline_frames",
            "strict_prior_available_frames", "strict_prior_S2_exceeds_baseline_frames")}, records=records,
        interpretation="Cached development diagnostic; full24 HOLD unchanged. Original BG/MW fields compare through first divergence; original actual-root S2 may differ. All-frame forecast counts include contexts after divergence. With no divergence, prior counts cover the complete common trajectory.",
        RNG_boundary="Outer prefix RNG equality follows from the shared initial planning seed, equal causal context/rates/window/sample count and prefix draw fees through first divergence; cached RNG snapshots are absent. Inner current/BG RNG states and native scores are not asserted equal.")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"summary": str(output), "pairs": 24, "divergent_pairs": result["divergent_pairs"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "md/experiment_artifacts")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summarize(args.artifact_dir, args.output or args.artifact_dir / (STEM + "_summary.json"))


if __name__ == "__main__":
    main()
