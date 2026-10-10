"""Change only BG planning randomness on the frozen steady/bursty tapes."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, process_time

from algorithm.background_competition_continuous_lane_rollout import BackgroundCompetitionContinuousLaneRollout
from algorithm.experiments.aligned_hardware_background_competition_continuous import background_work, check_rollout
from algorithm.experiments.aligned_hardware_bg_bursty_arrivals import (
    STEM as ORIGINAL, IDENTITY, PROFILES, read, save, require, births,
    load_protocol as load_original_protocol, performance_screen,
)
from algorithm.experiments.aligned_hardware_background_root_counterfactual import paired_outcomes
from algorithm.experiments.aligned_hardware_performance_necessity import load_models
from algorithm.experiments.aligned_hardware_s1_only_supplemental import ProgressSelector
from algorithm.experiments.bg_planning_stream_statistics import analyse_planning_streams
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory


ROOT = Path(__file__).resolve().parents[2]
STEM = "aligned_hardware_BG_planning_streams_20261004"
ORIGINAL_EPOCH = "2026-10-03T17:08:54.918852+00:00"
REPLICATIONS = (1, 2, 3)
PLANNING_SEED_RULE = "arrival-plan-sensitivity-v1:{node}:{load}:online:replicate{replication}"
UNIT_IDENTITY = ("unit_index", "profile", "planning_replication", "planning_seed", *IDENTITY)
EXTRA_SOURCES = (str(Path(__file__).resolve().relative_to(ROOT)),
    "algorithm/experiments/bg_planning_stream_statistics.py")


def path(directory, suffix):
    return Path(directory) / (STEM + suffix)


def original_path(directory, suffix):
    return Path(directory) / (ORIGINAL + suffix)


def cached_MW(directory, protocol, unit):
    """Read one authentic source MW arm; keep its original snapshot dimensions."""
    row = read(Path(directory) / unit["cached_MW_row_file"])
    saved = read(Path(directory) / unit["cached_MW_trajectory_file"])
    source_unit = unit["cached_MW_unit"]
    require(row["execution_checks_passed"] and saved["unit"] == source_unit
        and row["registered_at_utc"] == saved["registered_at_utc"] == protocol["source_execution_epoch"]
        and row["trajectory_file"] == unit["cached_MW_trajectory_file"]
        and all(row[key] == source_unit[key] for key in ("unit_index", "policy", "profile", *IDENTITY)),
        "cached MW identity/epoch/checks differ")
    return saved["result"]


def register(directory):
    started, cpu = perf_counter(), process_time()
    design = read(path(directory, "_design.json"))
    original = load_original_protocol(directory)
    completed = read(original_path(directory, "_summary.json"))
    require(original["registered_at_utc"] == completed["registered_at_utc"]
        == design["source_execution_epoch"] == ORIGINAL_EPOCH
        and completed["execution_checks_passed"] and completed["completed_new_trajectories"] == 768,
        "completed frozen steady/bursty experiment required")
    require(len(original["tapes"]) == 192 and original["profiles"] == list(PROFILES)
        and original["arrival_seeds"] == list(range(200001, 200013))
        and original["planning_samples"] == 8 and original["planning_event_cap"] == 128
        and original["birth_cohort"] == [300, 900] and original["arrival_horizon"] == 900
        and original["environment_arrival_end"] == original["observation_end"] == 1200
        and design["planning_replications"] == list(REPLICATIONS)
        and design["planning_seed_rule"] == PLANNING_SEED_RULE,
        "retain all192 tapes, original models/budget/clock and three frozen planning streams")
    mw_units = {unit["tape_id"]: unit for unit in original["units"] if unit["policy"] == "maxweight"}
    require(len(mw_units) == 192 and set(mw_units) == {tape["tape_id"] for tape in original["tapes"]},
        "all192 unique MW source arms required")
    epoch, units = datetime.now(timezone.utc).isoformat(), []
    source_protocol = dict(source_execution_epoch=ORIGINAL_EPOCH)
    for tape in original["tapes"]:
        source_unit = mw_units[tape["tape_id"]]
        require(all(source_unit[key] == tape[key] for key in ("profile", *IDENTITY)),
            "source tape and MW unit identities differ")
        fixture = read(Path(directory) / source_unit["fixture_file"])
        require(fixture["registered_at_utc"] == ORIGINAL_EPOCH and fixture["tape"] == tape
            and fixture["model"] == next(model for model in original["models"] if model["node"] == tape["node"]),
            "frozen source fixture differs")
        cached = dict(cached_MW_unit=source_unit,
            cached_MW_row_file=ORIGINAL + f"_unit_{source_unit['unit_index']:03d}.json",
            cached_MW_trajectory_file=ORIGINAL + f"_unit_{source_unit['unit_index']:03d}_trajectory.json.gz")
        mw = cached_MW(directory, source_protocol, cached)
        require(births(mw) == [(int(c), int(i), float(t)) for t, c, i in sorted(tape["arrivals"])],
            "cached MW actual births differ")
        for replication in REPLICATIONS:
            units.append(dict(unit_index=len(units), fixture_file=source_unit["fixture_file"], policy="BG",
                planning_replication=replication,
                planning_seed=PLANNING_SEED_RULE.format(node=tape["node"], load=tape["load"], replication=replication),
                profile=tape["profile"], **{key: tape[key] for key in IDENTITY}, **cached))
    require(len(units) == 576, "complete192-tape three-stream BG roster required")
    sources = {**original["execution_sources"], **{name: (ROOT / name).read_text() for name in EXTRA_SOURCES}}
    save(path(directory, "_protocol.json"), dict(experiment=STEM, registered_at_utc=epoch, design=design,
        source_execution_epoch=ORIGINAL_EPOCH, source_protocol_file=ORIGINAL + "_protocol.json",
        source_summary_file=ORIGINAL + "_summary.json", source_merge_correction=completed.get("merge_correction"),
        **{key: original[key] for key in ("models", "capacities", "loads", "performance", "planning_samples",
            "planning_event_cap", "clock", "birth_cohort", "arrival_horizon", "environment_arrival_end",
            "observation_end", "arrival_seeds", "profiles", "tapes", "paired_tape_checks")},
        planning_replications=REPLICATIONS, planning_seed_rule=PLANNING_SEED_RULE, units=units,
        execution_sources=sources, expected_new_trajectories=576, reused_MW_trajectory_arms=192,
        new_shadow_references=0, new_MW_simulations=0, statistical_scope=design["statistics"],
        limitations=design["limitations"],
        registration_work=dict(source_fixtures_read=192, cached_MW_trajectories_read=192,
            literal_tapes_generated=0, new_fixtures=0, wall_s=perf_counter() - started, cpu_s=process_time() - cpu)))
    print('{"registered":true,"scientific_units":576,"reused_MW_arms":192,"new_MW_simulations":0}', flush=True)


def load_protocol(directory):
    protocol = read(path(directory, "_protocol.json"))
    require(protocol["design"] == read(path(directory, "_design.json")), "registered design changed")
    for name, literal in protocol["execution_sources"].items():
        require((ROOT / name).read_text() == literal, "registered source changed: " + name)
    return protocol


def run_unit(directory, index):
    started, cpu = perf_counter(), process_time()
    protocol = load_protocol(directory)
    require(0 <= index < 576, "unit-index must be0..575")
    unit = protocol["units"][index]
    row_file, trajectory_file = (path(directory, f"_unit_{index:03d}{suffix}")
        for suffix in (".json", "_trajectory.json.gz"))
    require(not row_file.exists() and not trajectory_file.exists(), "preserve completed BG trajectory")
    fixture = read(Path(directory) / unit["fixture_file"])
    require(fixture["registered_at_utc"] == protocol["source_execution_epoch"]
        and all(fixture["tape"][key] == unit[key] for key in ("profile", *IDENTITY)), "source fixture identity differs")
    model, = load_models([fixture["model"]])
    tape = fixture["tape"]
    planner = BackgroundCompetitionContinuousLaneRollout(model.options, unit["planning_seed"], 8, 128)
    selector = ProgressSelector(planner)
    result = simulate_observed_rate_contextual_trajectory(model, tape["arrivals"], 1200, 1200, selector, policy="BG")
    save(trajectory_file, dict(registered_at_utc=protocol["registered_at_utc"], unit=unit, result=result))
    require(births(result) == [(int(c), int(i), float(t)) for t, c, i in sorted(tape["arrivals"])],
        "new BG actual births differ")
    require(check_rollout(result, model, protocol, planner), "frozen BG causal/budget/selection checks failed")
    work = background_work(result, planner, 900)
    work["controller_constructors"] = 1
    save(row_file, dict(**unit, registered_at_utc=protocol["registered_at_utc"], execution_checks_passed=True,
        trajectory_file=trajectory_file.name, work=work,
        computation=dict(selector_calls=selector.calls, selector_wall_s=selector.wall_s,
            max_selector_wall_s=selector.max_wall_s, wall_s=perf_counter() - started, cpu_s=process_time() - cpu,
            computation_delays_service=False)))
    print('{"unit_index":' + str(index) + ',"execution_checks_passed":true}', flush=True)


def merge(directory):
    started, cpu = perf_counter(), process_time()
    protocol = load_protocol(directory)
    rows = [read(path(directory, f"_unit_{unit['unit_index']:03d}.json")) for unit in protocol["units"]]
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
    save(path(directory, "_summary.json"), dict(experiment=STEM, registered_at_utc=protocol["registered_at_utc"],
        source_execution_epoch=protocol["source_execution_epoch"], execution_checks_passed=True,
        completed_new_trajectories=576, literal_tapes=192, reused_MW_trajectory_arms=192,
        new_MW_simulations=0, new_shadow_references=0, statistics=statistics, performance_screens=screens,
        totals_work=work, statistical_scope=protocol["statistical_scope"], limitations=protocol["limitations"],
        BG_main_result_preserved=True, automatic_adoption=False, best_stream_selection=False,
        full_trajectories_retained_on_server=True,
        merge_work=dict(new_BG_trajectory_records_read=576, cached_MW_trajectory_records_read=192,
            separately_recomputed_BG_MW_pairs=576, wall_s=perf_counter() - started, cpu_s=process_time() - cpu)))
    print('{"execution_checks_passed":true,"completed_new_trajectories":576,"new_MW_simulations":0}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "md/experiment_artifacts")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--unit-index", type=int)
    mode.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.register:
        register(args.artifact_dir)
    elif args.merge:
        merge(args.artifact_dir)
    else:
        run_unit(args.artifact_dir, args.unit_index)


if __name__ == "__main__":
    main()
