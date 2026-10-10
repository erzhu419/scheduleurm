"""Frozen development panel for BG under measured-phase-derived service."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, process_time

from algorithm.experiments.aligned_hardware_background_competition_continuous import background_work
from algorithm.experiments.aligned_hardware_background_root_counterfactual import paired_outcomes
from algorithm.experiments.aligned_hardware_bg_strong_controls import read, save, require, births
from algorithm.experiments.aligned_hardware_frame_experiment import poisson_arrivals
from algorithm.experiments.aligned_hardware_performance_necessity import load_models
from algorithm.experiments.aligned_hardware_s1_only_supplemental import ProgressSelector
from algorithm.experiments.bg_empirical_execution_checks import (
    ExactRecordedBG, check_empirical_execution, check_recorded_bg,
)
from algorithm.experiments.bg_empirical_service_statistics import analyse_empirical_development, performance_screen
from algorithm.experiments.empirical_service_lane_replay import (
    EmpiricalServiceTape, MeanService, simulate_empirical_service_trajectory,
)


ROOT = Path(__file__).resolve().parents[2]
STEM = "aligned_hardware_BG_empirical_service_development_20261004"
ORIGINAL = "aligned_hardware_background_competition_continuous_confirmation_20260930"
CONDITIONS = ("mean", "empirical")
POLICIES = ("maxweight", "BG")
IDENTITY = ("tape_id", "node", "lanes", "load", "seed", "service_seed")
UNIT_IDENTITY = ("unit_index", "condition", "policy", "planning_seed", "service_seed_key", *IDENTITY)
EXTRA_SOURCES = (str(Path(__file__).resolve().relative_to(ROOT)),
    "algorithm/experiments/bg_empirical_execution_checks.py",
    "algorithm/experiments/bg_empirical_service_statistics.py",
    "algorithm/experiments/empirical_service_lane_replay.py",
    "algorithm/experiments/aligned_hardware_empirical_service_samples.py",
    "algorithm/experiments/aligned_hardware_bg_strong_controls.py",
    "algorithm/experiments/aligned_hardware_s1_only_supplemental.py",
    "algorithm/experiments/aligned_hardware_s1_only_development.py",
    "algorithm/experiments/s1_only_supplemental_statistics.py",
    "algorithm/continuous_arrival_s1_only_lane_rollout.py",
    "algorithm/experiments/bg_bursty_statistics.py",
    "algorithm/experiments/bg_confirmation_block_intervals.py",
    "algorithm/experiments/aligned_hardware_background_root_counterfactual.py",
    "algorithm/experiments/single_action_counterfactual_replay.py")


def path(directory, suffix):
    return Path(directory) / (STEM + suffix)


def register(directory):
    started, cpu = perf_counter(), process_time()
    design_path = path(directory, "_design.json")
    design = read(design_path)
    bank_path = Path(directory) / design["sample_bank"]
    loads_path = Path(directory) / design["mean_capacity_calibration"]
    bank, loads = read(bank_path), read(loads_path)
    population, policy_design = design["population"], design["policy_and_information"]
    require(population["arrival_seeds"] == [300001, 300002, 300003]
        and population["service_seeds"] == [400001, 400002, 400003]
        and population["execution_conditions"] == list(CONDITIONS) and population["policies"] == list(POLICIES)
        and population["literal_arrival_tapes"] == 24 and population["new_trajectories"] == 96
        and population["birth_cohort"] == [300, 900] and population["cohort_checkpoint"] == 900
        and population["environment_arrival_end"] == population["observation_end"] == 1200
        and policy_design["planning_samples"] == 8 and policy_design["planning_event_cap"] == 128,
        "frozen three-block24-tape96-arm development population required")
    original = read(Path(directory) / (ORIGINAL + "_protocol.json"))
    for name, literal in original["execution_sources"].items():
        require((ROOT / name).read_text() == literal, "frozen BG source changed: " + name)
    models = {model["node"]: model for model in bank["mean_model_snapshots"]}
    sample_models = {model["node"]: model for model in bank["catalogues"]}
    require({(model["node"], model["lanes"]) for model in models.values()}
        == {tuple(item) for item in population["catalogue_lanes"]}
        and loads["same_rates_for_both_policies_and_execution_conditions"] is True
        and loads["sample_bank"] == bank_path.name, "frozen mean models and recalibrated shared rates required")
    epoch, tapes, units = datetime.now(timezone.utc).isoformat(), [], []
    for node, lanes in population["catalogue_lanes"]:
        model = models[node]
        samples_by_action = {action["action_id"]: action["training"] for action in sample_models[node]["actions"]}
        require(set(samples_by_action) == {option["action_id"] for option in model["options"]}
            and all(samples_by_action.values()), "every mean-catalogue action needs its accepted training support")
        for load in population["loads"]:
            calibration = next(row for row in loads["calibrations"] if row["node"] == node and row["load"] == load)
            require(calibration["lanes"] == lanes and calibration["support_slack_delta"] > 0,
                "strictly interior mean-renewal capacity calibration required")
            for seed, service_seed in zip(population["arrival_seeds"], population["service_seeds"], strict=True):
                index = len(tapes)
                tape = dict(tape_id=f"{node}:R{lanes}:load{load}:seed{seed}", node=node, lanes=lanes,
                    load=load, seed=seed, service_seed=service_seed,
                    arrivals=poisson_arrivals(calibration["arrival_rates"], 1200., seed))
                fixture = path(directory, f"_fixture_{index:03d}.json.gz")
                save(fixture, dict(registered_at_utc=epoch, model=model, tape=tape, samples_by_action=samples_by_action))
                tapes.append(tape)
                for condition in CONDITIONS:
                    for policy in POLICIES:
                        units.append(dict(unit_index=len(units), fixture_file=fixture.name, condition=condition,
                            policy=policy, planning_seed=policy_design["planning_seed_rule"].format(node=node, load=load),
                            service_seed_key=design["service_pairing"]["seed_rule"].format(
                                node=node, load=load, service_seed=service_seed), **{key: tape[key] for key in IDENTITY}))
    require(len(tapes) == 24 and len(units) == 96, "complete24-tape two-condition two-policy roster required")
    sources = {**original["execution_sources"], **{name: (ROOT / name).read_text() for name in EXTRA_SOURCES}}
    performance = dict(min_gain=design["performance_gates"]["mean_reduction"],
        group_secondary_tolerance=design["performance_gates"]["per_group_p90_and_unfinished_tolerance"])
    save(path(directory, "_protocol.json"), dict(experiment=STEM, phase="development", registered_at_utc=epoch,
        design=design, source_BG_execution_epoch=original["registered_at_utc"], models=list(models.values()),
        loads=population["loads"], calibrations=loads["calibrations"], tapes=tapes, units=units,
        arrival_seeds=population["arrival_seeds"], service_seeds=population["service_seeds"], conditions=CONDITIONS,
        policies=POLICIES, birth_cohort=population["birth_cohort"], arrival_horizon=900., observation_end=1200.,
        environment_arrival_end=1200., planning_samples=8, planning_event_cap=128,
        planning_seed_rule=policy_design["planning_seed_rule"], service_seed_rule=design["service_pairing"]["seed_rule"],
        performance=performance, clock=policy_design["clock"], execution_sources=sources,
        frozen_artifact_literals={file.name: file.read_text() for file in (design_path, bank_path, loads_path)},
        sample_bank_coverage=bank["coverage"], expected_new_trajectories=96, new_shadow_references=0,
        reused_trajectory_arms=0, statistics=design["statistics"], limitations=design["limitations"],
        validation="Literal births, empirical sample replay/FIFO/hidden-release execution and same-call exact BG score checks. Prior short operational mean-engine equivalence tests are not new shadow simulations or final performance evidence.",
        registration_work=dict(literal_tapes_generated=24, fixtures=24, wall_s=perf_counter() - started,
            cpu_s=process_time() - cpu)))
    print('{"registered":true,"scientific_units":96,"new_shadow_references":0}', flush=True)


def load_protocol(directory):
    protocol = read(path(directory, "_protocol.json"))
    for name, literal in protocol["frozen_artifact_literals"].items():
        require((Path(directory) / name).read_text() == literal, "registered design/data changed: " + name)
    for name, literal in protocol["execution_sources"].items():
        require((ROOT / name).read_text() == literal, "registered source changed: " + name)
    return protocol


def run_unit(directory, index):
    started, cpu = perf_counter(), process_time()
    protocol = load_protocol(directory)
    require(0 <= index < 96, "unit-index must be0..95")
    unit = protocol["units"][index]
    row_file, trajectory_file = (path(directory, f"_unit_{index:03d}{suffix}")
        for suffix in (".json", "_trajectory.json.gz"))
    require(not row_file.exists() and not trajectory_file.exists(), "preserve completed scientific trajectory")
    fixture = read(Path(directory) / unit["fixture_file"])
    require(fixture["registered_at_utc"] == protocol["registered_at_utc"]
        and all(fixture["tape"][key] == unit[key] for key in IDENTITY), "fixture identity/epoch differs")
    model, = load_models([fixture["model"]])
    planner = ExactRecordedBG(model.options, unit["planning_seed"], 8, 128)
    service = (MeanService() if unit["condition"] == "mean" else
        EmpiricalServiceTape(fixture["samples_by_action"], unit["service_seed_key"]))
    selector = ProgressSelector(planner.maxweight if unit["policy"] == "maxweight" else planner)
    result = simulate_empirical_service_trajectory(model, fixture["tape"]["arrivals"],
        protocol["environment_arrival_end"], protocol["observation_end"],
        selector, service, policy=unit["policy"])
    result["condition"], result["service_seed"] = unit["condition"], unit["service_seed_key"]
    save(trajectory_file, dict(registered_at_utc=protocol["registered_at_utc"], unit=unit, result=result))
    require(births(result) == [(int(c), int(i), float(t)) for t, c, i in sorted(fixture["tape"]["arrivals"])],
        "actual literal births differ")
    require(check_empirical_execution(result, model, protocol, service), "empirical execution/replayed service checks failed")
    if unit["policy"] == "BG":
        require(check_recorded_bg(result, model, protocol, planner), "recorded exact BG causal/budget/selection checks failed")
    work = background_work(result, planner, 900.)
    work["controller_constructors"] = 1
    exact_records = sum(bool(frame["rollout"].get("exact_root_scores")) for frame in result["frames"])
    save(row_file, dict(**unit, registered_at_utc=protocol["registered_at_utc"], execution_checks_passed=True,
        trajectory_file=trajectory_file.name, work=work,
        service_environment_work=dict(sampler_constructors=1,
            action_stream_initializations=len(model.options) if unit["condition"] == "empirical" else 0,
            support_record_draws=sum(service.counts.values()) if unit["condition"] == "empirical" else 0),
        exact_score_diagnostic_work=dict(exact_score_records=exact_records,
            retained_fraction_pairs=4 * exact_records, scores_recomputed=0),
        computation=dict(selector_calls=selector.calls, selector_wall_s=selector.wall_s,
            max_selector_wall_s=selector.max_wall_s, wall_s=perf_counter() - started, cpu_s=process_time() - cpu,
            computation_delays_service=False)))
    print('{"unit_index":' + str(index) + ',"execution_checks_passed":true}', flush=True)


def merge(directory):
    started, cpu = perf_counter(), process_time()
    protocol = load_protocol(directory)
    rows = [read(path(directory, f"_unit_{unit['unit_index']:03d}.json")) for unit in protocol["units"]]
    require(len(rows) == 96 and all(row["execution_checks_passed"]
        and row["registered_at_utc"] == protocol["registered_at_utc"]
        and all(row[key] == unit[key] for key in UNIT_IDENTITY)
        for row, unit in zip(rows, protocol["units"], strict=True)), "all96 prespecified units must pass")
    indexed = {(row["tape_id"], row["condition"], row["policy"]): row for row in rows}
    comparisons = {condition: [] for condition in CONDITIONS}
    models = {model["node"]: model for model in protocol["models"]}
    for tape in protocol["tapes"]:
        expected = [(int(c), int(i), float(t)) for t, c, i in sorted(tape["arrivals"])]
        for condition in CONDITIONS:
            arms = {}
            for policy in POLICIES:
                row = indexed[tape["tape_id"], condition, policy]
                saved = read(Path(directory) / row["trajectory_file"])
                require(saved["registered_at_utc"] == protocol["registered_at_utc"]
                    and all(saved["unit"][key] == row[key] for key in UNIT_IDENTITY), "saved scientific arm identity/epoch differs")
                require(saved["result"]["condition"] == condition and saved["result"]["service_seed"] == row["service_seed_key"]
                    and births(saved["result"]) == expected, "saved arm condition/service seed/births differ")
                arms[policy] = {key: saved["result"][key] for key in ("jobs", "snapshots")}
            outcomes = paired_outcomes(arms["maxweight"], arms["BG"], protocol, models[tape["node"]]["workloads"])
            comparisons[condition].append(dict(**{key: tape[key] for key in IDENTITY}, condition=condition, outcomes=outcomes))
    statistics = analyse_empirical_development(comparisons)
    work = {condition: {policy: Counter() for policy in POLICIES} for condition in CONDITIONS}
    environment_work = {condition: {policy: Counter() for policy in POLICIES} for condition in CONDITIONS}
    diagnostic_work = {condition: {policy: Counter() for policy in POLICIES} for condition in CONDITIONS}
    for row in rows:
        totals = work[row["condition"]][row["policy"]]
        for key, value in row["work"].items():
            totals[key] = max(totals[key], value) if key == "max_rollout_calls" else totals[key] + value
        environment_work[row["condition"]][row["policy"]].update(row["service_environment_work"])
        diagnostic_work[row["condition"]][row["policy"]].update(row["exact_score_diagnostic_work"])
    screens = {condition: performance_screen(statistics["condition_summaries"][condition], protocol["performance"])
        for condition in CONDITIONS}
    save(path(directory, "_summary.json"), dict(experiment=STEM, registered_at_utc=protocol["registered_at_utc"],
        phase="development", execution_checks_passed=True, completed_new_trajectories=96, literal_tapes=24,
        new_shadow_references=0, reused_trajectory_arms=0, statistics=statistics, performance_screens=screens,
        totals_work=work, service_environment_work=environment_work,
        exact_score_diagnostic_work=diagnostic_work,
        statistical_scope=protocol["statistics"], limitations=protocol["limitations"],
        automatic_adoption=False, BG_main_result_preserved=True, full_trajectories_retained_on_server=True,
        merge_work=dict(unit_rows_read=96, scientific_trajectory_records_read=96, separate_condition_policy_pairs=48,
            wall_s=perf_counter() - started, cpu_s=process_time() - cpu)))
    print('{"execution_checks_passed":true,"completed_new_trajectories":96,"new_shadow_references":0}', flush=True)


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
