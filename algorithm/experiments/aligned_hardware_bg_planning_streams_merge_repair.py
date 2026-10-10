"""Merge the frozen planning-stream roster using one separately validated row.

Unit462's saved trajectory is retained; its missing original PASS row remains
absent. The independent exact-Fraction certificate must be complete before
this analysis-only merge can use the separately named analysis record.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, process_time

from algorithm.experiments.aligned_hardware_bg_planning_streams import (
    ROOT, STEM, REPLICATIONS, UNIT_IDENTITY, IDENTITY, PROFILES, read, save,
    require, births, load_protocol, path, cached_MW, paired_outcomes,
    analyse_planning_streams, performance_screen,
)


def merge(directory):
    started, cpu = perf_counter(), process_time()
    analysis_registered_at_utc = datetime.now(timezone.utc).isoformat()
    protocol = load_protocol(directory)
    require(not path(directory, "_unit_462.json").exists(), "original failed-check row must remain absent")
    analysis_row_file = path(directory, "_analysis_unit_462.json")
    analysis_row = read(analysis_row_file)
    validation = analysis_row["analysis_validation"]
    require(validation["original_frozen_checker_passed"] is False
        and validation["exact_checker_passed"] is True, "independent unit462 validation required")
    proof = read(Path(directory) / validation["exact_proof_file"])
    require(proof["registered_at_utc"] == protocol["registered_at_utc"] and proof["unit_index"] == 462
        and proof["original_checker_passed"] is False and proof["corrected_checker_passed"] is True
        and proof["exact_strict_improvement_proved"] is True and proof["frozen_sources_unchanged"] is True
        and proof["all_other_frozen_checks_certified"] is True,
        "complete same-epoch exact-Fraction proof required before merging")
    rows = [analysis_row if unit["unit_index"] == 462 else
        read(path(directory, f"_unit_{unit['unit_index']:03d}.json")) for unit in protocol["units"]]
    require(len(rows) == 576 and all(row["execution_checks_passed"]
        and row["registered_at_utc"] == protocol["registered_at_utc"]
        and all(row[key] == unit[key] for key in UNIT_IDENTITY)
        for row, unit in zip(rows, protocol["units"], strict=True)), "all576 registered BG units must pass")
    comparisons = {profile: [] for profile in PROFILES}
    tapes = {tape["tape_id"]: tape for tape in protocol["tapes"]}
    models = {model["node"]: model for model in protocol["models"]}
    mw_cache = {}
    work = {profile: {str(replication): Counter() for replication in REPLICATIONS} for profile in PROFILES}
    for row in rows:
        saved = read(Path(directory) / row["trajectory_file"])
        require(saved["registered_at_utc"] == protocol["registered_at_utc"]
            and all(saved["unit"][key] == row[key] for key in UNIT_IDENTITY), "new BG trajectory identity/epoch differs")
        tape = tapes[row["tape_id"]]
        expected = [(int(c), int(i), float(t)) for t, c, i in sorted(tape["arrivals"])]
        if row["tape_id"] not in mw_cache:
            mw_cache[row["tape_id"]] = cached_MW(directory, protocol, row)
        mw, bg = mw_cache[row["tape_id"]], saved["result"]
        require(births(mw) == births(bg) == expected, "reused MW/new BG actual births differ")
        arms = [{"jobs": result["jobs"], "snapshots": result["snapshots"]} for result in (mw, bg)]
        outcomes = paired_outcomes(*arms, protocol, models[row["node"]]["workloads"])
        comparisons[row["profile"]].append(dict(**{key: row[key] for key in IDENTITY},
            profile=row["profile"], planning_replication=row["planning_replication"],
            planning_seed=row["planning_seed"], outcomes=outcomes))
        totals = work[row["profile"]][str(row["planning_replication"])]
        for key, value in row["work"].items():
            totals[key] = max(totals[key], value) if key == "max_rollout_calls" else totals[key] + value
    statistics = analyse_planning_streams(comparisons)
    profiles = statistics["profile_summaries"]
    screens = {profile: dict(averaged=performance_screen(profiles[profile]["averaged"], protocol["performance"]),
        replications={str(replication): performance_screen(profiles[profile]["replications"][str(replication)],
            protocol["performance"]) for replication in REPLICATIONS}) for profile in PROFILES}
    source_file = str(Path(__file__).resolve().relative_to(ROOT))
    correction = dict(unit_index=462, original_checker_passed=False, exact_checker_passed=True,
        analysis_unit_record=analysis_row_file.name, exact_proof_artifact=validation["exact_proof_file"],
        analysis_validation=validation, analysis_registered_at_utc=analysis_registered_at_utc,
        analysis_source_file=source_file, analysis_source_literal=(ROOT / source_file).read_text(),
        statistical_parameters_changed=False, policy_changed=False, recovery_new_simulations=0,
        original_PASS_row_created=False, original575_rows_unchanged=True,
        operational_receipts={
            "canonical_premerge_validation": STEM + "_runner_canonical_recovery_verification.json",
            "original_controller_state": STEM + "_runner_controller.json",
            "completion_and_launcher_accounting": STEM + "_completion_receipt.json"},
        accounting="576 prespecified unique scientific trajectories; canonical work excludes unmeasured duplicate-launch work. Exact diagnostic scoring is recorded separately in the proof. No claim of exactly576 total simulator executions.")
    save(path(directory, "_summary.json"), dict(experiment=STEM, registered_at_utc=protocol["registered_at_utc"],
        source_execution_epoch=protocol["source_execution_epoch"], execution_checks_passed=True,
        completed_new_trajectories=576, literal_tapes=192, reused_MW_trajectory_arms=192,
        new_MW_simulations=0, new_shadow_references=0, statistics=statistics, performance_screens=screens,
        totals_work=work, statistical_scope=protocol["statistical_scope"], limitations=protocol["limitations"],
        BG_main_result_preserved=True, automatic_adoption=False, best_stream_selection=False,
        full_trajectories_retained_on_server=True, merge_correction=correction,
        merge_work=dict(new_BG_trajectory_records_read=576, cached_MW_trajectory_records_read=192,
            separately_recomputed_BG_MW_pairs=576, wall_s=perf_counter() - started, cpu_s=process_time() - cpu)))
    print('{"execution_checks_passed":true,"completed_new_trajectories":576,"analysis_recovered_records":1,"recovery_new_simulations":0}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "md/experiment_artifacts")
    parser.add_argument("--merge", action="store_true", help="merge the existing576 registered trajectories")
    args = parser.parse_args()
    merge(args.artifact_dir)


if __name__ == "__main__":
    main()
