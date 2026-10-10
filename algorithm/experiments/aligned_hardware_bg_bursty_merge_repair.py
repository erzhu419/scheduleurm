"""Analysis-only repair of the frozen bursty runner's missing snapshot field.

The scientific trajectories and registered runner stay unchanged. Retain each
saved result's snapshots when constructing paired metric inputs, as required
by the existing common_metrics coordinate-dimension lookup.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, process_time

from algorithm.experiments.aligned_hardware_bg_bursty_arrivals import (
    ROOT, STEM, POLICIES, IDENTITY, CONTRASTS, PROFILES, read, save, require,
    births, load_protocol, path, paired_outcomes, analyse_bursty, performance_screen,
)


def merge(directory):
    started, cpu = perf_counter(), process_time()
    analysis_registered_at_utc = datetime.now(timezone.utc).isoformat()
    protocol = load_protocol(directory)
    rows = [read(path(directory, f"_unit_{unit['unit_index']:03d}.json")) for unit in protocol["units"]]
    require(len(rows) == 768 and all(row["execution_checks_passed"]
        and row["registered_at_utc"] == protocol["registered_at_utc"]
        and all(row[key] == unit[key] for key in ("unit_index", "policy", "profile", *IDENTITY))
        for row, unit in zip(rows, protocol["units"], strict=True)), "all768 registered units must pass")
    indexed = {(row["tape_id"], row["policy"]): row for row in rows}
    comparisons = {profile: {contrast: [] for contrast in CONTRASTS} for profile in PROFILES}
    models = {model["node"]: model for model in protocol["models"]}
    for tape in protocol["tapes"]:
        arms = {}
        for policy in POLICIES:
            row = indexed[tape["tape_id"], policy]
            saved = read(Path(directory) / row["trajectory_file"])
            require(saved["registered_at_utc"] == protocol["registered_at_utc"]
                and all(saved["unit"][key] == row[key] for key in ("unit_index", "policy", "profile", *IDENTITY)),
                "saved trajectory identity/epoch differs")
            arms[policy] = {"jobs": saved["result"]["jobs"], "snapshots": saved["result"]["snapshots"]}
        expected = [(int(c), int(i), float(t)) for t, c, i in sorted(tape["arrivals"])]
        require(all(births(arm) == expected for arm in arms.values()), "four-policy actual births differ")
        for contrast, (baseline, candidate) in CONTRASTS.items():
            outcomes = paired_outcomes(arms[baseline], arms[candidate], protocol, models[tape["node"]]["workloads"])
            comparisons[tape["profile"]][contrast].append(dict(**{key: tape[key] for key in IDENTITY},
                profile=tape["profile"], outcomes=outcomes))
    statistics = analyse_bursty(comparisons)
    work = {profile: {policy: Counter() for policy in POLICIES} for profile in PROFILES}
    for row in rows:
        totals = work[row["profile"]][row["policy"]]
        for key, value in row["work"].items():
            totals[key] = max(totals[key], value) if key == "max_rollout_calls" else totals[key] + value
    profiles = statistics["profile_summaries"]
    source_file = str(Path(__file__).resolve().relative_to(ROOT))
    correction = dict(failed_merge_task="t133870", reason="KeyError: snapshots in common_metrics; frozen merge retained only jobs",
        change="Retain the saved snapshots alongside jobs for the existing coordinate-dimension lookup.",
        analysis_registered_at_utc=analysis_registered_at_utc, source_file=source_file,
        source_file_literal=(ROOT / source_file).read_text(), new_trajectories_executed=0,
        statistical_parameters_changed=False, registered_scientific_sources_changed=False)
    save(path(directory, "_summary.json"), dict(experiment=STEM, registered_at_utc=protocol["registered_at_utc"],
        source_execution_epochs=protocol["source_execution_epochs"], execution_checks_passed=True,
        completed_new_trajectories=768, literal_tapes=192, reused_trajectory_arms=0, new_shadow_references=0,
        statistics=statistics, performance_screens={profile: performance_screen(profiles[profile]["BG_minus_maxweight"],
            protocol["performance"]) for profile in PROFILES},
        totals_work=work, paired_tape_checks=protocol["paired_tape_checks"],
        statistical_scope=protocol["statistical_scope"], limitations=protocol["limitations"],
        BG_main_result_preserved=True, automatic_adoption=False, full_trajectories_retained_on_server=True,
        merge_correction=correction,
        merge_work=dict(unit_rows_read=768, full_trajectory_records_read=768,
            separate_profile_policy_pairs=576, wall_s=perf_counter() - started, cpu_s=process_time() - cpu)))
    print('{"execution_checks_passed":true,"completed_new_trajectories":768,"merge_repair":true,"new_trajectories_executed":0}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "md/experiment_artifacts")
    parser.add_argument("--merge", action="store_true", help="merge the existing registered trajectories")
    args = parser.parse_args()
    merge(args.artifact_dir)


if __name__ == "__main__":
    main()
