"""Frozen S1-only opponent on all96 existing BG confirmation tapes."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, process_time

from algorithm.continuous_arrival_s1_only_lane_rollout import ContinuousArrivalS1OnlyLaneRollout
from algorithm.experiments.aligned_hardware_bg_strong_controls import (
    TimedSelector, births, read, require, same, save)
from algorithm.experiments.aligned_hardware_background_root_counterfactual import paired_outcomes
from algorithm.experiments.aligned_hardware_continuous_arrival_age_tail import cohort_metrics
from algorithm.experiments.aligned_hardware_observed_rate_age_tail_development import work_totals
from algorithm.experiments.aligned_hardware_performance_necessity import load_models
from algorithm.experiments.aligned_hardware_s1_only_development import check_s1_only, contrast_summary
from algorithm.experiments.arrival_aware_lane_rollout_summary import summarize
from algorithm.experiments.bg_confirmation_block_intervals import _validate_roster
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory
from algorithm.experiments.s1_only_supplemental_statistics import supplemental_statistics

ROOT = Path(__file__).resolve().parents[2]
STEM = "aligned_hardware_S1_only_supplemental_20261004"
BG = "aligned_hardware_background_competition_continuous_confirmation_20260930"
CONTROLS = "aligned_hardware_BG_strong_controls_20261003"
DEVELOPMENT = "aligned_hardware_S1_only_development_20261003"
CUTOFFS = ("arrival_horizon", "observation_end")
EXTRA_SOURCES = (
    str(Path(__file__).resolve().relative_to(ROOT)),
    "algorithm/continuous_arrival_s1_only_lane_rollout.py",
    "algorithm/experiments/aligned_hardware_s1_only_development.py",
    "algorithm/experiments/s1_only_supplemental_statistics.py",
    "algorithm/experiments/arrival_aware_lane_rollout_summary.py")
IDENTITY = ("tape_id", "node", "lanes", "load", "seed")


def path(directory, suffix):
    return Path(directory) / (STEM + suffix)


def register(directory):
    started, cpu = perf_counter(), process_time()
    design = read(path(directory, "_design.json"))
    base = read(Path(directory) / (BG + "_protocol.json"))
    reference = read(Path(directory) / (BG + ".json"))
    controls = read(Path(directory) / (CONTROLS + "_protocol.json"))
    controls_result = read(Path(directory) / (CONTROLS + "_summary.json"))
    development = read(Path(directory) / (DEVELOPMENT + "_protocol.json"))
    require(base["registered_at_utc"] == reference["registered_at_utc"]
        == controls["source_execution_epoch"] == design["source_execution_epochs"]["BG"]
        and controls["registered_at_utc"] == controls_result["registered_at_utc"]
        == design["source_execution_epochs"]["no_background"]
        and development["registered_at_utc"] == design["source_execution_epochs"]["S1_development"],
        "source execution epochs differ")
    require(reference["execution_checks_passed"] and controls_result["execution_checks_passed"]
        and controls_result["completed_new_trajectories"] == 192
        and reference["decision"] == "BACKGROUND_COMPETITION_CONTINUOUS_CONFIRMATION_PASS",
        "completed original BG and controls required")
    sources = {}
    for protocol in (base, controls, development):
        for name, literal in protocol["execution_sources"].items():
            require((ROOT / name).read_text() == literal, "frozen source changed: " + name)
            sources[name] = literal
    _validate_roster(base["tapes"])
    _validate_roster(reference["paired_runs"])
    shared = ("models", "loads", "arrival_seeds", "birth_cohort", "arrival_horizon",
              "environment_arrival_end", "observation_end", "planning_samples", "planning_event_cap", "clock")
    require(all(same(base[key], controls[key]) for key in shared)
        and base["planning_samples"] == 8 and base["planning_event_cap"] == 128
        and base["birth_cohort"] == [300, 900] and base["arrival_horizon"] == 900
        and base["environment_arrival_end"] == base["observation_end"] == 1200
        and same(base["models"], development["models"]), "retain frozen models, budget and clock")
    cached = read(Path(directory) / (BG + "_trajectories.json.gz"))
    require(cached["registered_at_utc"] == base["registered_at_utc"] and len(cached["runs"]) == 96,
        "all original BG/MW cached trajectories required")
    original = {row["tape_id"]: row["policies"] for row in cached["runs"]}
    references = {row["tape_id"]: row for row in reference["paired_runs"]}
    control_units = {unit["tape_id"]: unit for unit in controls["units"] if unit["policy"] == "no_background"}
    ids = {tape["tape_id"] for tape in base["tapes"]}
    require(len(control_units) == 96 and set(original) == set(references) == set(control_units) == ids,
        "same complete96 tape roster required")
    epoch, units = datetime.now(timezone.utc).isoformat(), []
    models = {model["node"]: model for model in base["models"]}
    for index, tape in enumerate(base["tapes"]):
        old, ref, control_unit = original[tape["tape_id"]], references[tape["tape_id"]], control_units[tape["tape_id"]]
        require(ref["baseline_reproduction_passed"] and ref["rollout_checks_passed"]
            and all(control_unit[key] == tape[key] for key in IDENTITY), "cached identity/checks differ")
        prefix = Path(directory) / (CONTROLS + f"_unit_{control_unit['unit_index']:03d}")
        row = read(str(prefix) + ".json")
        trajectory = read(str(prefix) + "_trajectory.json.gz")
        old_fixture = read(Path(directory) / control_unit["fixture_file"])
        require(row["execution_checks_passed"] and row["registered_at_utc"] == trajectory["registered_at_utc"]
            == controls["registered_at_utc"] and trajectory["unit"] == control_unit
            and all(row[key] == control_unit[key] for key in ("unit_index", "policy", *IDENTITY)),
            "no-background cached execution differs")
        require(old_fixture["registered_at_utc"] == controls["registered_at_utc"]
            and old_fixture["source_execution_epoch"] == base["registered_at_utc"]
            and same(old_fixture["tape"], tape) and same(old_fixture["model"], models[tape["node"]])
            and all(same(old_fixture["cached_policies"][label]["jobs"], old[label]["jobs"])
                    for label in ("maxweight", "arrival_rollout")),
            "original control fixture model/tape/BG/MW identities differ")
        no_background = trajectory["result"]
        require(births(old["maxweight"]) == births(old["arrival_rollout"]) == births(no_background)
            == [(int(c), int(i), float(t)) for t, c, i in sorted(tape["arrivals"])], "cached births differ")
        for cutoff in CUTOFFS:
            require(same(cohort_metrics(old, base[cutoff], base["birth_cohort"]), ref["snapshots"][cutoff]),
                "original BG/MW cohort metrics differ")
        workloads = tuple(models[tape["node"]]["workloads"])
        require(same(paired_outcomes(old["arrival_rollout"], no_background, base, workloads), row["outcomes"])
            and same(paired_outcomes(old["maxweight"], no_background, base, workloads), row["versus_MW"]),
            "original no-background paired metrics differ")
        fixture = path(directory, f"_fixture_{index:03d}.json.gz")
        arms = {"maxweight": old["maxweight"], "BG": old["arrival_rollout"], "no_background": no_background}
        save(fixture, dict(registered_at_utc=epoch, model=models[tape["node"]], tape=tape,
            source_execution_epochs=design["source_execution_epochs"],
            cached_policies={label: {"jobs": arm["jobs"], "snapshots": arm["snapshots"]} for label, arm in arms.items()},
            cached_MW_work=ref["work"]["maxweight"]))
        units.append(dict(unit_index=index, fixture_file=fixture.name, **{key: tape[key] for key in IDENTITY}))
    sources.update({name: (ROOT / name).read_text() for name in EXTRA_SOURCES})
    save(path(directory, "_protocol.json"), dict(experiment=STEM, registered_at_utc=epoch,
        design=design, source_execution_epochs=design["source_execution_epochs"],
        **{key: base[key] for key in (*shared, "planning_seed_rule", "performance", "tapes")},
        units=units, execution_sources=sources, expected_new_trajectories=96,
        reused_BG_MW_no_background_trajectory_arms=288, new_bg_mw_or_no_background_simulations=0,
        new_shadow_references=0, statistical_scope=design["statistics"],
        limitations=design["statistics"],
        cached_totals_work={"BG_MW": reference["totals_work"],
            "no_background": controls_result["policies"]["no_background"]["totals_work"]},
        registration_work=dict(cached_trajectory_records_read=288, fixtures=96,
            wall_s=perf_counter() - started, cpu_s=process_time() - cpu)))
    print('{"registered":true,"scientific_units":96,"new_reference_simulations":0}', flush=True)


def load_protocol(directory):
    protocol = read(path(directory, "_protocol.json"))
    require(protocol["design"] == read(path(directory, "_design.json")), "registered design changed")
    for name, literal in protocol["execution_sources"].items():
        require((ROOT / name).read_text() == literal, "registered source changed: " + name)
    return protocol


class ProgressSelector(TimedSelector):
    """Publish observed service-clock progress without changing policy inputs."""
    def __init__(self, planner):
        super().__init__(planner)
        self.started, self.last_log = perf_counter(), 0.

    def __call__(self, *context):
        decision = super().__call__(*context)
        elapsed = perf_counter() - self.started
        if self.calls == 1 or elapsed - self.last_log >= 45:
            print('SCHEDULEURM_PROGRESS {"current":' + str(context[3])
                + ',"total":1200,"unit":"service_seconds","elapsed_seconds":' + str(elapsed) + '}', flush=True)
            self.last_log = elapsed
        return decision


def run_unit(directory, index):
    started, cpu = perf_counter(), process_time()
    protocol = load_protocol(directory)
    require(0 <= index < 96, "unit-index must be0..95")
    unit = protocol["units"][index]
    row_file, trajectory_file = (path(directory, f"_unit_{index:03d}{suffix}")
        for suffix in (".json", "_trajectory.json.gz"))
    require(not row_file.exists() and not trajectory_file.exists(), "preserve completed unit")
    fixture = read(Path(directory) / unit["fixture_file"])
    require(fixture["registered_at_utc"] == protocol["registered_at_utc"]
        and fixture["source_execution_epochs"] == protocol["source_execution_epochs"]
        and fixture["tape"]["tape_id"] == unit["tape_id"], "fixture identity/epoch differs")
    model, = load_models([fixture["model"]])
    tape, cached = fixture["tape"], fixture["cached_policies"]
    planner = ContinuousArrivalS1OnlyLaneRollout(model.options,
        f"arrival-plan-v1:{model.node}:{tape['load']}:online", 8, 128)
    selector = ProgressSelector(planner)
    result = simulate_observed_rate_contextual_trajectory(model, tape["arrivals"], 1200, 1200, selector, policy="S1_only")
    save(trajectory_file, dict(registered_at_utc=protocol["registered_at_utc"], unit=unit, result=result))
    require(all(births(result) == births(arm) for arm in cached.values()), "new trajectory births differ")
    require(check_s1_only(result, model, protocol, planner), "S1-only causal/budget/selection checks failed")
    contrasts = {label: paired_outcomes(arm, result, protocol, model.workloads) for label, arm in cached.items()}
    work = work_totals(result, planner, 900)
    work["controller_constructors"] = 1
    for key in ("diagnostic_S2_comparisons", "root_ordering_comparisons", "strict_improvement_comparisons"):
        work[key] = sum(frame["rollout"][key] for frame in result["frames"])
    work["diagnostic_S2_exceeding_decisions"] = sum(frame["rollout"]["chosen_exceeds_MW_S2"] for frame in result["frames"])
    save(row_file, dict(**unit, registered_at_utc=protocol["registered_at_utc"], execution_checks_passed=True,
        trajectory_file=trajectory_file.name, contrasts=contrasts, work=work, cached_MW_work=fixture["cached_MW_work"],
        computation=dict(selector_calls=selector.calls, selector_wall_s=selector.wall_s,
            max_selector_wall_s=selector.max_wall_s, wall_s=perf_counter() - started, cpu_s=process_time() - cpu,
            computation_delays_service=False)))
    print('{"unit_index":' + str(index) + ',"execution_checks_passed":true}', flush=True)


def merge(directory):
    started, cpu = perf_counter(), process_time()
    protocol = load_protocol(directory)
    rows = [read(path(directory, f"_unit_{unit['unit_index']:03d}.json")) for unit in protocol["units"]]
    _validate_roster(rows)
    require(all(row["execution_checks_passed"] and row["registered_at_utc"] == protocol["registered_at_utc"]
        and all(row[key] == unit[key] for key in ("unit_index", *IDENTITY))
        for row, unit in zip(rows, protocol["units"], strict=True)), "all96 registered units must pass")
    gate_rows = [{**{key: row[key] for key in IDENTITY}, "baseline_reproduction_passed": True,
        "rollout_checks_passed": row["execution_checks_passed"],
        "snapshots": {cutoff: {"maxweight": arms["factual"], "arrival_rollout": arms["replacement"]}
            for cutoff, arms in row["contrasts"]["maxweight"]["snapshots"].items()},
        "work": {"maxweight": row["cached_MW_work"], "arrival_rollout": row["work"]}} for row in rows]
    totals = Counter()
    for row in rows:
        for key, value in row["work"].items():
            totals[key] = max(totals[key], value) if key == "max_rollout_calls" else totals[key] + value
    screen = summarize(gate_rows, protocol)
    screen["statistical_scope"] = (
        "Descriptive original MW-reference screen on the96 reused confirmation tapes and12 seed blocks; "
        "post-confirmation supplemental comparison conditional on fixed planning streams, no new adoption gate.")
    save(path(directory, "_summary.json"), dict(experiment=STEM, registered_at_utc=protocol["registered_at_utc"],
        source_execution_epochs=protocol["source_execution_epochs"], execution_checks_passed=True,
        completed_new_trajectories=96, reused_BG_MW_no_background_trajectory_arms=288,
        new_bg_mw_or_no_background_simulations=0, new_shadow_references=0,
        comparisons={label: contrast_summary(rows, label) for label in ("no_background", "BG", "maxweight")},
        statistics=supplemental_statistics(rows), MW_reference_screen=screen,
        totals_work=dict(totals), cached_totals_work=protocol["cached_totals_work"],
        statistical_scope=protocol["statistical_scope"], BG_main_result_preserved=True,
        automatic_adoption=False, full_trajectories_retained_on_server=True,
        merge_work=dict(unit_rows_read=96, block_interval_calls=6,
            wall_s=perf_counter() - started, cpu_s=process_time() - cpu)))
    print('{"execution_checks_passed":true,"completed_new_trajectories":96}', flush=True)


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
